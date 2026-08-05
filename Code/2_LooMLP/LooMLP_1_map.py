import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import random
import warnings
import matplotlib as mpl
import math

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ================================================
DATA_PATH = '../../Data/LooMLP/Data_1_demo.csv'
OUTPUT_PATH = '../../Data/LooMLP/Data_2_LooM_demo(not applicable).csv'
VIZ_OUTPUT_DIR = '../../Data/LooMLP/LooM_Curve_demo'
CURVE_DATA_DIR = f"{VIZ_OUTPUT_DIR}/curve_data"
RULES_OUTPUT_PATH = f"{VIZ_OUTPUT_DIR}/LooM_rule.csv"
BATCH_SIZE = 1024
LR = 0.004
EPOCHS = 2
HIDDEN_DIM = 64
WEIGHT_DECAY = 1e-4
SEED = 42
PATIENCE = 1

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{VIZ_OUTPUT_DIR}/models", exist_ok=True)
os.makedirs(CURVE_DATA_DIR, exist_ok=True)


class RuleDataset(Dataset):
    def __init__(self, df, sim_col, sent_col):
        self.data = {
            "sims": torch.tensor(df[sim_col].values, dtype=torch.float32).reshape(-1, 1),
            "sents": torch.tensor(df[sent_col].values, dtype=torch.float32).reshape(-1, 1)
        }
        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "sims": self.data["sims"][idx],
            "sents": self.data["sents"][idx]
        }

class LooMMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.base_magnitude = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.sentiment_modulation = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )

        self.modulation_strength = nn.Parameter(torch.tensor(0.1))

        self.sign_factor = 5.0

        self.zero_suppress = nn.Parameter(torch.tensor(3.0))
        self.output_scale = nn.Parameter(torch.tensor(0.3))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, sims, sents):
        sims = sims.reshape(-1, 1) if len(sims.shape) == 1 else sims
        sents = sents.reshape(-1, 1) if len(sents.shape) == 1 else sents

        sign = torch.tanh(self.sign_factor * sents)

        base_magnitude = torch.abs(self.base_magnitude(sims))

        sent_abs = torch.abs(sents)
        modulation_raw = self.sentiment_modulation(sent_abs)
        modulation_normalized = torch.sigmoid(modulation_raw)
        modulation_range = 1.0 + self.modulation_strength * (modulation_normalized - 0.5) * 2

        magnitude = torch.abs(base_magnitude * modulation_range)

        zero_region = torch.exp(-self.zero_suppress * (sims ** 2 * 2.0 + sent_abs ** 2 * 0.5))
        output = magnitude * sign * (1.0 - zero_region)
        final_output = torch.tanh(output * self.output_scale)

        return final_output.squeeze(-1)

    def forward_for_rules(self, inputs):
        return self.forward(inputs[:, 0:1], inputs[:, 1:2])

class AdaptiveLooMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.rule_weights = nn.Parameter(torch.ones(4))
        self.progress_scale = nn.Parameter(torch.ones(4))

    def update_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, pred, sims, sents, model, data_stats=None):
        if len(sims.shape) > 1:
            sims = sims.reshape(-1)
        if len(sents.shape) > 1:
            sents = sents.reshape(-1)
        if len(pred.shape) > 1:
            pred = pred.reshape(-1)

        progress = min(self.epoch / (EPOCHS - 1), 1.0)

        if data_stats is None:
            sim_min, sim_max = 0.0, 1.0
            sent_min, sent_max = -1.0, 1.0
            low_sim_thresh = 0.05
            neutral_sent_thresh = 0.05
            high_sim_thresh = 0.9
        else:
            sim_min, sim_max = data_stats["sim_min"], data_stats["sim_max"]
            sent_min, sent_max = data_stats["sent_min"], data_stats["sent_max"]
            low_sim_thresh = 0.05
            neutral_sent_thresh = 0.05
            high_sim_thresh = 0.9

        pos_sent_mask = sents > 0
        neg_sent_mask = sents < 0
        low_sim_mask = sims < low_sim_thresh
        neutral_sent_mask = torch.abs(sents) < neutral_sent_thresh
        high_sim_mask = sims > high_sim_thresh
        high_pos_sent_mask = sents > neutral_sent_thresh
        high_neg_sent_mask = sents < -neutral_sent_thresh

        zero_condition_mask = low_sim_mask | neutral_sent_mask
        high_sim_pos_sent_mask = high_sim_mask & high_pos_sent_mask
        high_sim_neg_sent_mask = high_sim_mask & high_neg_sent_mask

        weight_inputs = self.rule_weights + self.progress_scale * progress
        all_weights = F.softplus(weight_inputs)
        rule1_weight, rule2_weight, rule3_weight, rule4_weight = all_weights

        pos_preds = pred.masked_select(pos_sent_mask)
        neg_preds = pred.masked_select(neg_sent_mask)
        sign_loss_pos = F.relu(-pos_preds).mean() if pos_preds.numel() > 0 else torch.tensor(0.0).to(pred.device)
        sign_loss_neg = F.relu(neg_preds).mean() if neg_preds.numel() > 0 else torch.tensor(0.0).to(pred.device)
        rule1_loss = (sign_loss_pos + sign_loss_neg) * rule1_weight

        zero_preds = pred.masked_select(zero_condition_mask)
        rule2_loss = (zero_preds ** 2).mean() * rule2_weight if zero_preds.numel() > 0 else torch.tensor(0.0).to(
            pred.device)

        delta_sim = (sim_max - sim_min) * 0.1
        delta_sent_abs = (sent_max - sent_min) * 0.05

        batch_size = len(sims)
        sample_size = min(batch_size // 2 + 50, batch_size)

        mono_loss = torch.tensor(0.0).to(pred.device)

        if sample_size > 0:
            g = torch.Generator(device='cpu')
            g.manual_seed(SEED + self.epoch)
            sample_idx = torch.randperm(batch_size, generator=g)[:sample_size]

            sims_sample = sims[sample_idx]
            sents_sample = sents[sample_idx]

            inputs_base = torch.stack([sims_sample, sents_sample], dim=1)
            pred_base = model.forward_for_rules(inputs_base)

            sims_delta = torch.clamp(sims_sample + delta_sim, sim_min, sim_max)
            inputs_delta_sim = torch.stack([sims_delta, sents_sample], dim=1)

            sent_signs = torch.sign(sents_sample)
            sent_abs = torch.abs(sents_sample)
            delta_sent = delta_sent_abs * sent_signs
            sents_delta = torch.clamp(sents_sample + delta_sent, sent_min, sent_max)

            inputs_delta_sent = torch.stack([sims_sample, sents_delta], dim=1)

            pred_delta_sim = model.forward_for_rules(inputs_delta_sim)
            pred_delta_sent = model.forward_for_rules(inputs_delta_sent)

            non_neutral_mask_sample = (torch.abs(sents_sample) >= neutral_sent_thresh)
            high_sim_mask_sample = (sims_sample >= low_sim_thresh)
            valid_samples = non_neutral_mask_sample & high_sim_mask_sample

            if valid_samples.any():
                valid_sims = sims_sample[valid_samples]
                expected_delta_sim = 0.01 * (1.0 + valid_sims)

                sim_delta_abs = torch.abs(pred_delta_sim[valid_samples]) - torch.abs(pred_base[valid_samples])
                sim_mono_loss = F.relu(expected_delta_sim - sim_delta_abs).mean()

                valid_sents = sents_sample[valid_samples]
                expected_delta_sent = 0.01 * (1.0 + torch.abs(valid_sents))

                sent_delta_abs = torch.abs(pred_delta_sent[valid_samples]) - torch.abs(pred_base[valid_samples])
                sent_mono_loss = F.relu(expected_delta_sent - sent_delta_abs).mean()

                mono_loss = (sim_mono_loss + sent_mono_loss) * rule3_weight

        boundary_loss = torch.tensor(0.0).to(pred.device)

        if high_sim_pos_sent_mask.any():
            pos_pred = pred.masked_select(high_sim_pos_sent_mask)
            pos_target = torch.ones_like(pos_pred)
            boundary_loss += ((pos_pred - pos_target) ** 2).mean()

        if high_sim_neg_sent_mask.any():
            neg_pred = pred.masked_select(high_sim_neg_sent_mask)
            neg_target = -torch.ones_like(neg_pred)
            boundary_loss += ((neg_pred - neg_target) ** 2).mean()

        if high_sim_pos_sent_mask.any() or high_sim_neg_sent_mask.any():
            boundary_loss = boundary_loss * rule4_weight

        total_loss = rule1_loss + rule2_loss + mono_loss + boundary_loss

        if torch.isnan(total_loss):
            total_loss = torch.tensor(1.0).to(pred.device)

        self.rule1_loss = rule1_loss
        self.rule2_loss = rule2_loss
        self.mono_loss = mono_loss
        self.boundary_loss = boundary_loss
        self.total_loss = total_loss

        return total_loss

def visualize_loom_surface_optimized(model, sim_col, data_stats, save_dir=VIZ_OUTPUT_DIR, return_data=False):
    model.eval()
    with torch.no_grad():
        n_points = 150

        sim_min, sim_max = data_stats["sim_min"], data_stats["sim_max"]
        sent_min, sent_max = data_stats["sent_min"], data_stats["sent_max"]

        sim_values = np.linspace(sim_min, sim_max, n_points)
        sent_values = np.linspace(sent_min, sent_max, n_points)

        sim_grid = np.zeros((n_points, n_points))
        sent_grid = np.zeros((n_points, n_points))

        sim_grid = np.broadcast_to(sim_values.reshape(1, -1), (n_points, n_points))
        sent_grid = np.broadcast_to(sent_values.reshape(-1, 1), (n_points, n_points))

        sent_percentages = [-100, -66, -33, 0, 33, 66, 100]
        sent_values_target = []
        for percentage in sent_percentages:
            if percentage < 0:
                target_val = max(sent_min, sent_min * (1 - (percentage + 100) / 100))
            elif percentage > 0:
                target_val = min(sent_max, sent_max * percentage / 100)
            else:
                target_val = 0.0
            sent_values_target.append(target_val)

        sim_percentages = [0, 20, 40, 60, 80, 100]
        sim_values_target = []
        for percentage in sim_percentages:
            target_val = sim_min + (sim_max - sim_min) * percentage / 100
            target_val = np.clip(target_val, sim_min, sim_max)
            sim_values_target.append(target_val)

        grid_inputs = np.stack([sim_grid.flatten(), sent_grid.flatten()], axis=1)

        grid_tensor = torch.FloatTensor(grid_inputs).to(device)
        batch_size = BATCH_SIZE * 2

        predictions = []
        for i in range(0, len(grid_tensor), batch_size):
            batch = grid_tensor[i:i + batch_size]
            pred_batch = model(batch[:, 0:1], batch[:, 1:2]).cpu().numpy()
            predictions.append(pred_batch)

        loom_grid = np.concatenate(predictions).reshape(sim_grid.shape)

        if return_data:
            surface_data = {
                'sim_values': sim_values.tolist(),
                'sent_values': sent_values.tolist(),
                'sim_grid': sim_grid.tolist(),
                'sent_grid': sent_grid.tolist(),
                'loom_grid': loom_grid.tolist()
            }

        base_filename = f"{save_dir}/{sim_col.replace('/', '_').replace(' ', '_')}"

        plt.figure(figsize=(7, 5))
        ax = plt.subplot(111, projection='3d')
        surf = ax.plot_surface(sim_grid, sent_grid, loom_grid, cmap='coolwarm', vmin=-1, vmax=1)
        ax.view_init(elev=30, azim=225)
        ax.set_xlabel('Similarity')
        ax.set_ylabel('Sentiment')
        ax.set_zlabel('LooM')
        ax.set_zlim(-1.1, 1.1)
        plt.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='LooM')
        plt.tight_layout()
        plt.savefig(f"{base_filename}_3d.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{base_filename}_3d.pdf", dpi=600, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 5))
        contour = plt.contourf(sim_grid, sent_grid, loom_grid, 15, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='LooM')

        if np.min(loom_grid) < 0 < np.max(loom_grid):
            zero_contour = plt.contour(sim_grid, sent_grid, loom_grid, levels=[0], colors='black', linewidths=1.5)
            plt.clabel(zero_contour, inline=True, fontsize=8, fmt='%.1f')

        levels = np.linspace(-0.8, 0.8, 9)
        contour_lines = plt.contour(sim_grid, sent_grid, loom_grid, levels=levels, colors='white', linewidths=0.8)
        plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

        plt.xlabel('Similarity')
        plt.ylabel('Sentiment')
        plt.tight_layout()
        plt.savefig(f"{base_filename}_contour.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{base_filename}_contour.pdf", dpi=600, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(7, 5))
        cmap = plt.cm.coolwarm
        colors = [cmap(c) for c in np.linspace(0, 1, len(sent_percentages))]

        sim_template = torch.FloatTensor(sim_values).view(-1, 1).to(device)

        for i, (percentage, target_sent_val) in enumerate(zip(sent_percentages, sent_values_target)):
            sents_tensor = torch.full_like(sim_template, float(target_sent_val))

            batch_size = BATCH_SIZE * 2
            loom_values = []
            for j in range(0, len(sim_template), batch_size):
                sims_batch = sim_template[j:j + batch_size]
                sents_batch = sents_tensor[j:j + batch_size]
                pred_batch = model(sims_batch, sents_batch).cpu().numpy()
                loom_values.append(pred_batch)

            loom_values = np.concatenate(loom_values)

            linestyle = '--' if percentage == 0 else '-'
            linewidth = 2.5 if percentage == 0 else 2
            label = f'Sent={percentage}% ({target_sent_val:.2f})' if percentage != 0 else f'Sent=0%'

            plt.plot(sim_values, loom_values, color=colors[i], linewidth=linewidth,
                     linestyle=linestyle, label=label)

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
        plt.grid(alpha=0.3)
        plt.ylim(-1.1, 1.1)
        plt.xlabel('Similarity')
        plt.ylabel('LooM')
        plt.legend(loc='best', fontsize='small', framealpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{base_filename}_sim_curves.png", dpi=600)
        plt.savefig(f"{base_filename}_sim_curves.pdf", dpi=600)
        plt.close()

        plt.figure(figsize=(7, 5))
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(sim_percentages)))

        sent_template = torch.FloatTensor(sent_values).view(-1, 1).to(device)

        for i, (percentage, target_sim_val) in enumerate(zip(sim_percentages, sim_values_target)):
            sims_tensor = torch.full_like(sent_template, float(target_sim_val))

            batch_size = BATCH_SIZE * 2
            loom_values = []
            for j in range(0, len(sent_template), batch_size):
                sents_batch = sent_template[j:j + batch_size]
                sims_batch = sims_tensor[j:j + batch_size]
                pred_batch = model(sims_batch, sents_batch).cpu().numpy()
                loom_values.append(pred_batch)

            loom_values = np.concatenate(loom_values)

            linestyle = '--' if percentage == 0 else '-'
            linewidth = 2.5 if percentage == 0 else 2
            label = f'Sim={percentage}% ({target_sim_val:.2f})' if percentage != 0 else f'Sim=0%'

            plt.plot(sent_values, loom_values, linestyle=linestyle, linewidth=linewidth,
                     color=colors[i], label=label)

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
        plt.ylim(-1.1, 1.1)
        plt.xlim(sent_min, sent_max)
        plt.grid(alpha=0.3)
        plt.xlabel('Sentiment')
        plt.ylabel('LooM')
        plt.legend(loc='best', fontsize='small', framealpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{base_filename}_sent_curves.png", dpi=600)
        plt.savefig(f"{base_filename}_sent_curves.pdf", dpi=600)
        plt.close()

        if return_data:
            return surface_data
        else:
            return None

def export_curve_data_optimized(models, sim_cols, data_stats_dict, save_dir=CURVE_DATA_DIR):
    all_sent_curves = {}
    all_sim_curves = {}
    all_surface_data = {}

    for sim_col in tqdm(sim_cols, desc="Export curve"):
        if sim_col not in models or sim_col not in data_stats_dict:
            continue

        model = models[sim_col]
        data_stats = data_stats_dict[sim_col]
        model.eval()

        try:
            surface_data = visualize_loom_surface_optimized(model, sim_col, data_stats, save_dir=VIZ_OUTPUT_DIR,
                                                            return_data=True)
            if surface_data:
                all_surface_data[sim_col] = surface_data
            else:
                n_points = 101
                sim_min_viz, sim_max_viz = data_stats["sim_min"], data_stats["sim_max"]
                sent_min_viz, sent_max_viz = data_stats["sent_min"], data_stats["sent_max"]
                sim_values_plot_default = np.linspace(sim_min_viz, sim_max_viz, n_points)
                sent_values_plot_default = np.linspace(sent_min_viz, sent_max_viz, n_points)
        except Exception as e:
            n_points = 101
            sim_min_viz, sim_max_viz = data_stats["sim_min"], data_stats["sim_max"]
            sent_min_viz, sent_max_viz = data_stats["sent_min"], data_stats["sent_max"]
            sim_values_plot_default = np.linspace(sim_min_viz, sim_max_viz, n_points)
            sent_values_plot_default = np.linspace(sent_min_viz, sent_max_viz, n_points)

        with torch.no_grad():
            sim_min, sim_max = data_stats["sim_min"], data_stats["sim_max"]
            sent_min, sent_max = data_stats["sent_min"], data_stats["sent_max"]

            sent_percentages = [-100, -66, -33, 0, 33, 66, 100]
            sent_values_target = []
            for percentage in sent_percentages:
                if percentage < 0:
                    target_val = max(sent_min, sent_min * (1 - (percentage + 100) / 100))
                elif percentage > 0:
                    target_val = min(sent_max, sent_max * percentage / 100)
                else:
                    target_val = 0.0
                sent_values_target.append(target_val)
            sent_values_target = np.array(sent_values_target)

            sim_percentages = [0, 20, 40, 60, 80, 100]
            sim_values_target = []
            for percentage in sim_percentages:
                target_val = sim_min + (sim_max - sim_min) * percentage / 100
                target_val = np.clip(target_val, sim_min, sim_max)
                sim_values_target.append(target_val)
            sim_values_target = np.array(sim_values_target)

            if surface_data and 'sim_values' in surface_data:
                sim_values_plot = np.array(surface_data['sim_values'])
            else:
                sim_values_plot = sim_values_plot_default

            sim_values_plot = np.unique(np.concatenate([sim_values_plot, sim_values_target]))

            if surface_data and 'sent_values' in surface_data:
                sent_values_plot = np.array(surface_data['sent_values'])
            else:
                sent_values_plot = sent_values_plot_default

            sent_values_plot = np.unique(np.concatenate([sent_values_plot, sent_values_target]))

            sims_tensor_plot = torch.FloatTensor(sim_values_plot).view(-1, 1).to(device)
            sents_tensor_plot = torch.FloatTensor(sent_values_plot).view(-1, 1).to(device)

            batch_size = BATCH_SIZE * 4
            sent_curves_export = {}

            for percentage, target_sent_val in zip(sent_percentages, sent_values_target):
                sents_tensor_slice = torch.full_like(sims_tensor_plot, float(target_sent_val))

                loom_values_slice = []
                for j in range(0, len(sims_tensor_plot), batch_size):
                    end_idx = min(j + batch_size, len(sims_tensor_plot))
                    pred_batch = model(
                        sims_tensor_plot[j:end_idx],
                        sents_tensor_slice[j:end_idx]
                    ).cpu().numpy()
                    loom_values_slice.append(pred_batch)

                loom_values_slice = np.concatenate(loom_values_slice)

                sent_curves_export[percentage] = {
                    'percentage': percentage,
                    'target_value': float(target_sent_val),
                    'curve': loom_values_slice.tolist()
                }

            all_sent_curves[sim_col] = {
                'sim_values': sim_values_plot.tolist(),
                'sent_percentages': sent_percentages,
                'sent_min': float(sent_min),
                'sent_max': float(sent_max),
                'curves': sent_curves_export
            }

            sim_curves_export = {}
            for percentage, target_sim_val in zip(sim_percentages, sim_values_target):
                sims_tensor_slice = torch.full_like(sents_tensor_plot, float(target_sim_val))

                loom_values_slice = []
                for j in range(0, len(sents_tensor_plot), batch_size):
                    end_idx = min(j + batch_size, len(sents_tensor_plot))
                    pred_batch = model(
                        sims_tensor_slice[j:end_idx],
                        sents_tensor_plot[j:end_idx]
                    ).cpu().numpy()
                    loom_values_slice.append(pred_batch)

                loom_values_slice = np.concatenate(loom_values_slice)
                key = f'sim_{percentage}'
                sim_curves_export[key] = {
                    'percentage': percentage,
                    'target_value': float(target_sim_val),
                    'curve': loom_values_slice.tolist()
                }

            all_sim_curves[sim_col] = {
                'sent_values': sent_values_plot.tolist(),
                'sim_percentages': sim_percentages,
                'sim_min': float(sim_min),
                'sim_max': float(sim_max),
                'curves': sim_curves_export
            }

    try:
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(f"{save_dir}/all_sent_curves.npz", **all_sent_curves)
        np.savez_compressed(f"{save_dir}/all_sim_curves.npz", **all_sim_curves)

        if all_surface_data:
            np.savez_compressed(f"{save_dir}/all_surface_data.npz", **all_surface_data)
        else:
            print("WARNING")

        print(f"Export {save_dir}")

    except Exception as e:
        print(f"Wrong: {e}")

    return {
        'all_sent_curves': all_sent_curves,
        'all_sim_curves': all_sim_curves,
        'all_surface_data': all_surface_data
    }

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

def train_model_optimized(sim_col, train_df, val_df, sent_col, data_stats, epochs=EPOCHS):
    train_dataset = RuleDataset(train_df, sim_col, sent_col)
    val_dataset = RuleDataset(val_df, sim_col, sent_col)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=False,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=False,
        num_workers=0
    )
    model = LooMMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = AdaptiveLooMLoss().to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR / 10)

    patience = PATIENCE
    counter = 0
    best_loss = float('inf')
    best_state = None

    print(f"Start training {sim_col} (epochs={epochs})...")

    for epoch in range(epochs):
        criterion.update_epoch(epoch)

        model.train()
        total_loss = 0.0
        batch_count = 0
        print(f"Epoch {epoch + 1}/{epochs} training...")

        for batch in train_loader:
            sims = batch["sims"].to(device)
            sents = batch["sents"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(sims, sents)
            loss = criterion(pred, sims, sents, model, data_stats)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count}/{len(train_loader)}, loss: {loss.item():.4f}", end="\r")

        print()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sims = batch["sims"].to(device)
                sents = batch["sents"].to(device)
                pred = model(sims, sents)
                loss = criterion(pred, sims, sents, model, data_stats)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs} Done: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            counter = 0
            print(f"Found new model，Val Loss: {best_loss:.4f}")
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping, {counter} no epoch improvement")
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model_path = f"{VIZ_OUTPUT_DIR}/models/{sim_col.replace('/', '_').replace(' ', '_')}_best.pth"
    torch.save(best_state, model_path)

    visualize_loom_surface_optimized(model, sim_col, data_stats)
    print(f"{sim_col} Train and vis done")

    return model_path, model, best_loss

def predict_optimized(model, df, sim_col, sent_col, data_stats=None, collect_losses=False):
    model.eval()

    sims = df[sim_col].values.astype(np.float32).reshape(-1, 1)
    sents = df[sent_col].values.astype(np.float32).reshape(-1, 1)

    batch_size = BATCH_SIZE
    n_samples = len(sims)
    all_preds = []
    rule_losses = None

    if collect_losses:
        criterion = AdaptiveLooMLoss().to(device)
        criterion.update_epoch(EPOCHS - 1)
        rule_losses = {
            "sign_loss": 0,
            "zero_condition_loss": 0,
            "monotonicity_loss": 0,
            "boundary_loss": 0,
            "total_loss": 0
        }

    all_sims = torch.FloatTensor(sims)
    all_sents = torch.FloatTensor(sents)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            sims_batch = all_sims[i:end_idx].to(device)
            sents_batch = all_sents[i:end_idx].to(device)

            pred = model(sims_batch, sents_batch)
            if collect_losses:
                loss = criterion(pred, sims_batch.squeeze(-1) if sims_batch.shape[-1] == 1 else sims_batch,
                                 sents_batch.squeeze(-1) if sents_batch.shape[-1] == 1 else sents_batch,
                                 model, data_stats)
                rule_losses["sign_loss"] += getattr(criterion, 'rule1_loss', torch.tensor(0)).item()
                rule_losses["zero_condition_loss"] += getattr(criterion, 'rule2_loss', torch.tensor(0)).item()
                rule_losses["monotonicity_loss"] += getattr(criterion, 'mono_loss', torch.tensor(0)).item()
                rule_losses["boundary_loss"] += getattr(criterion, 'boundary_loss', torch.tensor(0)).item()
                rule_losses["total_loss"] += getattr(criterion, 'total_loss', loss).item()

            all_preds.append(pred.cpu().numpy())
    result = np.concatenate(all_preds).flatten()
    if collect_losses:
        batch_count = math.ceil(n_samples / batch_size)
        for k in rule_losses:
            rule_losses[k] /= batch_count
        return result, rule_losses

    return result

def train_model_optimized(sim_col, train_df, val_df, sent_col, data_stats, epochs=EPOCHS):
    train_dataset = RuleDataset(train_df, sim_col, sent_col)
    val_dataset = RuleDataset(val_df, sim_col, sent_col)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=False,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=False,
        num_workers=0
    )

    model = LooMMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = AdaptiveLooMLoss().to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR / 10)

    patience = PATIENCE
    counter = 0
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        criterion.update_epoch(epoch)

        model.train()
        total_loss = 0.0

        for batch in train_loader:
            sims = batch["sims"].to(device)
            sents = batch["sents"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(sims, sents)
            loss = criterion(pred, sims, sents, model, data_stats)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sims = batch["sims"].to(device)
                sents = batch["sents"].to(device)
                pred = model(sims, sents)
                loss = criterion(pred, sims, sents, model, data_stats)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model_path = f"{VIZ_OUTPUT_DIR}/models/{sim_col.replace('/', '_').replace(' ', '_')}_best.pth"
    torch.save(best_state, model_path)

    visualize_loom_surface_optimized(model, sim_col, data_stats)

    return model_path, model, best_loss

def main_optimized():
    import time
    start_time = time.time()
    df = pd.read_csv(DATA_PATH, dtype={'Sentiment': np.float32})
    exclude_cols = ["Sentiment"]
    sim_cols = [col for col in df.columns if col not in exclude_cols]
    sent_col = "Sentiment"

    print(f"Data: {len(df)} line, Cols: {len(sim_cols)}")
    for col in sim_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
        df[col] = df[col].clip(0, 1)

    df[sent_col] = pd.to_numeric(df[sent_col], errors='coerce').fillna(0).astype(np.float32)
    df[sent_col] = df[sent_col].clip(-1, 1)
    df['_emotion_group'] = pd.qcut(df[sent_col], 5, labels=False, duplicates='drop')

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=SEED,
        stratify=df['_emotion_group']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED,
        stratify=temp_df['_emotion_group']
    )

    train_df = train_df.drop('_emotion_group', axis=1)
    val_df = val_df.drop('_emotion_group', axis=1)
    test_df = test_df.drop('_emotion_group', axis=1)
    df = df.drop('_emotion_group', axis=1)

    print(f"Train: {len(train_df)} line, Val: {len(val_df)} line, Test: {len(test_df)} line")

    data_stats_dict = {}
    for sim_col in sim_cols:
        sim_values = train_df[sim_col].values
        sent_values = train_df[sent_col].values

        data_stats_dict[sim_col] = {
            "sim_min": max(0.0, np.min(sim_values)),
            "sim_max": min(1.0, np.max(sim_values)),
            "sim_mean": np.mean(sim_values),
            "sim_75pct": np.percentile(sim_values, 75),
            "sent_min": max(-1.0, np.min(sent_values)),
            "sent_max": min(1.0, np.max(sent_values)),
            "sent_mean": np.mean(sent_values),
            "sent_75pct": np.percentile(sent_values, 75)
        }

    rules_losses = []
    models_dict = {}
    all_predictions = {}

    print(f"Start training {len(sim_cols)} LooM models...")
    for idx, sim_col in enumerate(tqdm(sim_cols, desc="Processing")):
        try:
            data_stats = data_stats_dict[sim_col]
            model_path, model, best_loss = train_model_optimized(
                sim_col, train_df, val_df, sent_col, data_stats
            )
            models_dict[sim_col] = model
            test_predictions, test_rule_losses = predict_optimized(
                model, test_df, sim_col, sent_col, data_stats, collect_losses=True
            )
            predictions, rule_losses = predict_optimized(
                model, df, sim_col, sent_col, data_stats, collect_losses=True
            )
            all_predictions[f"{sim_col}"] = predictions
            rule_loss_record = {
                "feature": sim_col,
                "规则1_符号一致": rule_losses["sign_loss"],
                "规则2_零点行为": rule_losses["zero_condition_loss"],
                "规则3_幅度单调性": rule_losses["monotonicity_loss"],
                "规则4_边界保真度": rule_losses["boundary_loss"],
                "总损失": rule_losses["total_loss"],
                "测试集总损失": test_rule_losses["total_loss"]
            }
            rules_losses.append(rule_loss_record)

        except Exception as e:
            print(f"✗ {sim_col} Wrong: {str(e)}")
            continue

    export_curve_data_optimized(models_dict, list(models_dict.keys()), data_stats_dict)
    rules_df = pd.DataFrame(rules_losses)
    rules_df.to_csv(RULES_OUTPUT_PATH, index=False, encoding='utf-8-sig')

    for sim_col in all_predictions:
        df[sim_col] = all_predictions[sim_col]

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    processed_features = len(models_dict)
    total_features = len(sim_cols)
    print(
        f"\nDone: {processed_features}/{total_features} cols ({processed_features / total_features * 100:.1f}%)")

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time: {int(hours)}h{int(minutes)}min{seconds:.1f}s")

if __name__ == "__main__":
    import time
    main_optimized()