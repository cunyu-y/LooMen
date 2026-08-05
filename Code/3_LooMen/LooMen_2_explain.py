import os, gc, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from captum.attr import IntegratedGradients
from tqdm import tqdm

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────

class Config:
    FILEPATH         = "../../Results/LooMen"
    MODEL_NAME       = "D:/Model/chinese-roberta-wwm-ext"

    FEAT_BASELINE    = "mean"
    MODEL_PATH       = f"{FILEPATH}/LooM.pth"
    CSV_FILE         = f"{FILEPATH}/test-LooM.csv"
    OUTPUT_DIR       = f"{FILEPATH}/IG_explain_demo"
    NON_FEATURE_COLS = ['Depression', 'Year', 'Gender', 'Covid', 'Text', 'City', 'Sentiment']
    TEXT_COL         = 'Text'
    LABEL_COL        = 'Depression'

    NUM_HEADS        = 8
    FEAT_DIM         = 88
    SAMPLE_SIZE      = 500000
    ENCODE_BATCH     = 512
    IG_SAMPLE_BATCH  = 512
    N_STEPS          = 50
    MAX_LEN          = 181
    TOP_N            = 20
    SEED             = 42
    AXIOM_CHECK_N    = 1000
    COMPLETENESS_TOL = 0.05


# ─────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

def load_data(cfg):
    data      = pd.read_csv(cfg.CSV_FILE)
    feat_cols = [c for c in data.columns if c not in cfg.NON_FEATURE_COLS]
    print(f"[Data] total {len(data)} line, with {len(feat_cols)} cols")
    return data, feat_cols

def sample_data(data, feat_cols, cfg):
    n   = len(data)
    idx = np.random.choice(n, cfg.SAMPLE_SIZE, replace=False) if cfg.SAMPLE_SIZE < n else np.arange(n)
    texts    = data[cfg.TEXT_COL].values[idx].tolist()
    features = data[feat_cols].values[idx].astype(np.float32)
    labels   = data[cfg.LABEL_COL].values[idx] if cfg.LABEL_COL in data.columns else None
    cities   = data['城市'].values[idx] if '城市' in data.columns else np.array([])
    print(f"[Sample] {len(texts)} data")
    return texts, features, labels, idx, cities

def sort_by_length(texts, tokenizer, max_len):
    lengths      = [len(tokenizer.encode(str(t), max_length=max_len, truncation=True)) for t in texts]
    sorted_order = np.argsort(lengths)[::-1].copy()
    return sorted_order

def build_feat_baseline(cfg, features):
    if   cfg.FEAT_BASELINE == "zero":   bl = np.zeros(features.shape[1], dtype=np.float32)
    elif cfg.FEAT_BASELINE == "mean":   bl = features.mean(axis=0).astype(np.float32)
    elif cfg.FEAT_BASELINE == "median": bl = np.median(features, axis=0).astype(np.float32)
    else: raise ValueError(f"unknown FEAT_BASELINE: {cfg.FEAT_BASELINE}")
    return bl


# ─────────────────────────────────────────────
class CrossAttentionLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = q_dim // num_heads
        self.q_proj    = nn.Linear(q_dim,  q_dim)
        self.k_proj    = nn.Linear(kv_dim, q_dim)
        self.v_proj    = nn.Linear(kv_dim, q_dim)
        self.out_proj  = nn.Linear(q_dim,  q_dim)
        self.dropout   = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(q_dim)
        self.norm2     = nn.LayerNorm(q_dim)
        self.ffn = nn.Sequential(
            nn.Linear(q_dim, q_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(q_dim * 4, q_dim), nn.Dropout(dropout)
        )

    def forward(self, q_emb, kv_emb):
        b       = q_emb.size(0)
        q       = self.q_proj(q_emb.unsqueeze(1)).view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k       = self.k_proj(kv_emb.unsqueeze(1)).view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v       = self.v_proj(kv_emb.unsqueeze(1)).view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn    = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        context = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, -1)
        attended = self.norm1(q_emb + self.dropout(self.out_proj(context)))
        return self.norm2(attended + self.ffn(attended))


class DualStream(nn.Module):
    def __init__(self, model_name, num_labels, feat_dim, num_heads):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        h_dim     = self.base.config.hidden_size

        self.text_proc = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(h_dim, h_dim), nn.LeakyReLU(), nn.LayerNorm(h_dim)
        )

        def make_feat_proc():
            return nn.Sequential(
                nn.BatchNorm1d(feat_dim),
                nn.Linear(feat_dim, feat_dim), nn.LeakyReLU(),
                nn.Linear(feat_dim, feat_dim), nn.GELU(),
                nn.LayerNorm(feat_dim)
            )

        def make_feat_proc_aux():
            return nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, feat_dim), nn.LeakyReLU(),
                nn.Linear(feat_dim, feat_dim), nn.GELU(),
                nn.LayerNorm(feat_dim)
            )

        self.feat_proc     = make_feat_proc()
        self.feat_proc_aux = make_feat_proc_aux()

        self.feat_proj_up   = nn.Linear(feat_dim, h_dim)
        self.feat_proj_down = nn.Linear(h_dim, feat_dim)
        self.cross_attn1    = CrossAttentionLayer(h_dim, h_dim, num_heads)
        self.cross_attn2    = CrossAttentionLayer(h_dim, h_dim, num_heads)

        self.joint_classifier = nn.Sequential(
            nn.Linear(h_dim + h_dim, 512), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )
        self.text_classifier = nn.Sequential(
            nn.Linear(h_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )
        self.feat_classifier = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )

        self.log_var_joint = nn.Parameter(torch.zeros(1))
        self.log_var_text  = nn.Parameter(torch.zeros(1))
        self.log_var_feat  = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def encode_text_batch(self, input_ids, attention_mask):
        cls = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.text_proc(cls).float()


def load_model(cfg, device):
    model = DualStream(cfg.MODEL_NAME, 2, cfg.FEAT_DIM, cfg.NUM_HEADS)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device), strict=True)
    model.to(device).eval()
    print("[Model] Loading successful")
    return model


def precompute_text_embs(texts, tokenizer, model, device, cfg):
    N             = len(texts)
    sorted_order  = sort_by_length(texts, tokenizer, cfg.MAX_LEN)
    restore_order = np.argsort(sorted_order)
    sorted_texts  = [texts[i] for i in sorted_order]
    h_dim         = model.base.config.hidden_size
    embs_sorted   = torch.zeros(N, h_dim, dtype=torch.float32)

    print("[Embedding] Calculating embedding...")
    for start in tqdm(range(0, N, cfg.ENCODE_BATCH), desc="Text embedding"):
        batch_texts = sorted_texts[start: start + cfg.ENCODE_BATCH]
        enc = tokenizer(batch_texts, return_tensors='pt', truncation=True,
                        padding=True, max_length=cfg.MAX_LEN).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            emb = model.encode_text_batch(enc['input_ids'], enc['attention_mask'])
        embs_sorted[start: start + len(batch_texts)] = emb.cpu()

    embs = embs_sorted[restore_order]
    print(f"[Embedding] Done，shape={embs.shape}")
    return embs


# ─────────────────────────────────────────────

def forward_feat(model, features, text_embs):
    feat_emb = model.feat_proc_aux(features)
    logits   = model.feat_classifier(feat_emb)
    return logits.float()[:, 1]

def make_forward_feat(model):
    def forward(features, text_embs):
        return forward_feat(model, features, text_embs)
    return forward

def make_joint_proba(model, device):
    def proba(features, text_embs):
        feat_emb = model.feat_proc(features)
        feat_up  = model.feat_proj_up(feat_emb)
        feat_up  = model.cross_attn1(feat_up, text_embs)
        feat_up  = model.cross_attn2(feat_up, text_embs)
        logits   = model.joint_classifier(torch.cat([feat_up, text_embs], dim=1)).float()
        return torch.softmax(logits, dim=1)[:, 1]
    return proba


# ─────────────────────────────────────────────

def verify_axioms(model, text_embs_all, features, feat_baseline, cfg, device):
    print(f"\n[Axiomatic val] Sample {cfg.AXIOM_CHECK_N} data...")
    forward_fn = make_forward_feat(model)
    ig         = IntegratedGradients(forward_fn)
    check_idx  = np.random.choice(len(features), min(cfg.AXIOM_CHECK_N, len(features)), replace=False)

    comp_errors, sens_results = [], []

    for idx in tqdm(check_idx, desc="Axiomatic val"):
        feat_t = torch.tensor(features[idx],   dtype=torch.float32, device=device).unsqueeze(0)
        base_t = torch.tensor(feat_baseline,   dtype=torch.float32, device=device).unsqueeze(0)
        text_t = text_embs_all[idx].unsqueeze(0).to(device)

        attr = ig.attribute(inputs=feat_t, baselines=base_t,
                            additional_forward_args=(text_t,),
                            n_steps=cfg.N_STEPS,
                            internal_batch_size=cfg.N_STEPS)
        attr_np = attr.squeeze(0).detach().cpu().float().numpy()

        with torch.no_grad():
            fx    = forward_fn(feat_t, text_t).item()
            fx_bl = forward_fn(base_t, text_t).item()
        comp_errors.append(abs(attr_np.sum() - (fx - fx_bl)))

        diff_dims = np.where(np.abs(features[idx] - feat_baseline) > 1e-8)[0]
        if len(diff_dims) == 0:
            continue
        for k in np.random.choice(diff_dims, min(5, len(diff_dims)), replace=False):
            feat_mod   = features[idx].copy(); feat_mod[k] = feat_baseline[k]
            feat_mod_t = torch.tensor(feat_mod, dtype=torch.float32, device=device).unsqueeze(0)
            attr_mod   = ig.attribute(inputs=feat_mod_t, baselines=base_t,
                                      additional_forward_args=(text_t,),
                                      n_steps=cfg.N_STEPS, internal_batch_size=cfg.N_STEPS)
            sens_results.append(abs(attr_mod.squeeze(0)[k].item()) < 1e-4)

    comp_arr  = np.array(comp_errors)
    comp_pass = float(np.mean(comp_arr < cfg.COMPLETENESS_TOL))
    sens_pass = float(np.mean(sens_results)) if sens_results else float('nan')

    result = {
        "completeness_mean_abs_error": round(float(np.mean(comp_arr)), 6),
        "completeness_max_abs_error":  round(float(np.max(comp_arr)),  6),
        "completeness_pass_rate":      round(comp_pass, 4),
        "completeness_tol":            cfg.COMPLETENESS_TOL,
        "sensitivity_pass_rate":       round(sens_pass, 4) if not np.isnan(sens_pass) else "nan",
        "sensitivity_n_checks":        len(sens_results),
        "n_checked":                   len(check_idx),
    }

    print(f"\n── Axiomatic validation results ──────────────────────────────────────────")
    print(f"  Completeness mean error         : {result['completeness_mean_abs_error']:.6f}")
    print(f"  Completeness max error          : {result['completeness_max_abs_error']:.6f}")
    print(f"  Completeness PASS（error<{cfg.COMPLETENESS_TOL}）  : {comp_pass*100:.1f}%")
    if sens_results:
        print(f"  Sensitivity PASS（|IG[k]|<1e-4）: {sens_pass*100:.1f}%（{len(sens_results)} times）")
    print("──────────────────────────────────────────────────────────\n")
    return result


# ─────────────────────────────────────────────

def compute_ig_global(cfg, model, text_embs_all, features, feat_baseline, device):
    N, feat_dim = features.shape
    ig_array    = np.zeros((N, feat_dim), dtype=np.float32)
    pred_scores = np.zeros(N, dtype=np.float32)
    forward_fn  = make_forward_feat(model)
    proba_fn    = make_joint_proba(model, device)
    ig          = IntegratedGradients(forward_fn)

    print(f"[IG] Calculating（batch={cfg.IG_SAMPLE_BATCH}, steps={cfg.N_STEPS}）...")
    for start in tqdm(range(0, N, cfg.IG_SAMPLE_BATCH), desc="Integrated Gradients"):
        end    = min(start + cfg.IG_SAMPLE_BATCH, N)
        b_feat = torch.tensor(features[start:end], dtype=torch.float32, device=device)
        b_text = text_embs_all[start:end].to(device)
        b_base = torch.tensor(np.tile(feat_baseline, (end - start, 1)), dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_scores[start:end] = proba_fn(b_feat, b_text).cpu().float().numpy()

        try:
            attr = ig.attribute(inputs=b_feat, baselines=b_base,
                                additional_forward_args=(b_text,),
                                n_steps=cfg.N_STEPS,
                                internal_batch_size=len(b_feat) * cfg.N_STEPS)
            ig_array[start:end] = attr.detach().cpu().float().numpy()
        except Exception as e:
            print(f"[Warn] batch [{start}:{end}] fail, back: {e}")
            for i in range(end - start):
                try:
                    attr_i = ig.attribute(
                        inputs=b_feat[i:i+1], baselines=b_base[i:i+1],
                        additional_forward_args=(b_text[i:i+1],),
                        n_steps=cfg.N_STEPS, internal_batch_size=cfg.N_STEPS)
                    ig_array[start + i] = attr_i.squeeze(0).detach().cpu().float().numpy()
                except Exception as e2:
                    print(f"  [Warn] Sample {start+i} fail: {e2}")

        if (start // cfg.IG_SAMPLE_BATCH + 1) % 20 == 0:
            gc.collect(); torch.cuda.empty_cache()

    gc.collect(); torch.cuda.empty_cache()
    print("[IG] Done")
    return ig_array, pred_scores


# ─────────────────────────────────────────────

def save_results(cfg, ig_array, feat_cols, features, labels, sample_idx, axiom_result, cities, pred_scores):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    mean_ig    = ig_array.mean(axis=0)
    summary_df = (pd.DataFrame({"feature": feat_cols, "mean_IG_class1": mean_ig})
                    .sort_values("mean_IG_class1", ascending=False))

    summary_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "Global_IG.csv"),
                      index=False, encoding='utf-8-sig')
    pd.DataFrame([axiom_result]).to_csv(os.path.join(cfg.OUTPUT_DIR, "Axon_results.csv"),
                                        index=False, encoding='utf-8-sig')

    top    = summary_df.head(cfg.TOP_N)
    bottom = summary_df.tail(cfg.TOP_N)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)
    axes[0].barh(top["feature"][::-1],  top["mean_IG_class1"][::-1],  color='crimson')
    axes[0].set_title(f"Top {cfg.TOP_N} Features (Mean IG)")
    axes[0].set_xlabel("Mean IG Attribution")
    axes[1].barh(bottom["feature"],     bottom["mean_IG_class1"],      color='steelblue')
    axes[1].set_title(f"Bottom {cfg.TOP_N} Features (Mean IG)")
    axes[1].set_xlabel("Mean IG Attribution")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "Global_IG.png"))
    plt.close()

    np.savez(os.path.join(cfg.OUTPUT_DIR, "IG_results.npz"),
             feature_ig_array=ig_array,
             feature_names=np.array(feat_cols),
             feature_values=features,
             mean_ig_class1=mean_ig,
             true_label=labels if labels is not None else np.array([]),
             sample_index=sample_idx,
             city=cities,
             y_pred_proba=pred_scores,
             feat_baseline_type=cfg.FEAT_BASELINE,
             axiom_comp_mean_abs=axiom_result["completeness_mean_abs_error"],
             axiom_comp_pass=axiom_result["completeness_pass_rate"],
             axiom_sens_pass=str(axiom_result["sensitivity_pass_rate"]))

    print(f"[SAVE] All results → {cfg.OUTPUT_DIR}")


# ─────────────────────────────────────────────

def main():
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.SEED)
    print(f"[Device] {device}")

    data, feat_cols                              = load_data(cfg)
    texts, features, labels, sample_idx, cities = sample_data(data, feat_cols, cfg)
    tokenizer                                    = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    model                                        = load_model(cfg, device)
    feat_baseline                                = build_feat_baseline(cfg, features)

    text_embs_all           = precompute_text_embs(texts, tokenizer, model, device, cfg)
    axiom_result            = verify_axioms(model, text_embs_all, features, feat_baseline, cfg, device)
    ig_array, pred_scores   = compute_ig_global(cfg, model, text_embs_all, features, feat_baseline, device)
    save_results(cfg, ig_array, feat_cols, features, labels, sample_idx, axiom_result, cities, pred_scores)


if __name__ == "__main__":
    main()