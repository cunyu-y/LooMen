import os
import glob
import time
import shutil
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (Trainer, TrainingArguments, AutoModel, AutoTokenizer,
                          EarlyStoppingCallback, DataCollatorWithPadding)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
SEED             = 42
DATA_PATH        = '../../Data/LooMen/Data_1_LooM_demo.csv'
FILEPATH         = "../../Results/LooMen"
TEST_CSV         = f'{FILEPATH}/test-LooM.csv'
MODEL_SAVE_PATH  = f'{FILEPATH}/LooM.pth'
RESULT_FILE_PATH = f'{FILEPATH}/Test_results.txt'

MODEL_NAME    = "D:/Model/chinese-roberta-wwm-ext"
NUM_EPOCHS    = 4
BATCH_SIZE    = 32
NUM_HEADS     = 8
LEARNING_RATE = 2e-5
THRESHOLD     = 0.26

NON_FEATURE_COLS = ['Depression', 'Year', 'Gender', 'Covid', 'Text', 'City', 'Sentiment']
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(FILEPATH, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)


def load_data(path):
    data = pd.read_csv(path)
    data = data.drop_duplicates(subset='Text', keep='first').reset_index(drop=True)
    return data


def split_by_sample(data):
    labels  = data['Depression'].tolist()
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels)
    train_idx, val_idx  = train_test_split(train_idx, test_size=0.2, random_state=SEED,
                                           stratify=[labels[i] for i in train_idx])
    print(f"--> Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


class LooMDataset(Dataset):
    def __init__(self, texts, labels, features, tokenizer, max_len):
        self.texts, self.labels, self.features = texts, labels, features
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, max_length=self.max_len,
                             padding=False, return_attention_mask=True, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
            'features':       torch.tensor(self.features[idx], dtype=torch.float)
        }


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
        b = q_emb.size(0)
        q = self.q_proj(q_emb.unsqueeze(1)).view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_emb.unsqueeze(1)).view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_emb.unsqueeze(1)).view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
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

    def encode_feat(self, features, use_aux=False):
        proc = self.feat_proc_aux if use_aux else self.feat_proc
        return proc(features)

    def forward(self, input_ids, attention_mask, features, labels=None):
        text_emb = self.text_proc(
            self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        )
        feat_emb      = self.encode_feat(features, use_aux=False)
        feat_emb_pure = self.encode_feat(features, use_aux=True)
        feat_up = self.feat_proj_up(feat_emb)
        feat_up = self.cross_attn1(feat_up, text_emb)
        feat_up = self.cross_attn2(feat_up, text_emb)
        joint_logits = self.joint_classifier(torch.cat([feat_up, text_emb], dim=1))
        text_logits  = self.text_classifier(text_emb)
        feat_logits  = self.feat_classifier(feat_emb_pure)

        if labels is not None:
            ce         = nn.CrossEntropyLoss()
            loss_joint = ce(joint_logits, labels)
            loss_text  = ce(text_logits,  labels)
            loss_feat  = ce(feat_logits,  labels)
            loss = (loss_joint / (2 * self.log_var_joint.exp()) + self.log_var_joint / 2
                  + loss_text  / (2 * self.log_var_text.exp())  + self.log_var_text  / 2
                  + loss_feat  / (2 * self.log_var_feat.exp())  + self.log_var_feat  / 2)
            return {'loss': loss, 'logits': joint_logits}

        return {'logits': joint_logits}


def compute_metrics(p):
    probs = torch.softmax(torch.tensor(p.predictions), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    return {
        'accuracy':  (preds == p.label_ids).mean(),
        'precision': precision_score(p.label_ids, preds, zero_division=0, average='macro'),
        'recall':    recall_score(p.label_ids, preds, zero_division=0, average='macro'),
        'f1':        f1_score(p.label_ids, preds, zero_division=0, average='macro'),
        'roc_auc':   roc_auc_score(p.label_ids, probs[:, 1]) if len(np.unique(p.label_ids)) > 1 else 0.0
    }


def evaluate_with_progress(model, dataset, tokenizer):
    model.eval()
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE,
                            collate_fn=DataCollatorWithPadding(tokenizer))
    all_probs, all_labels, total_loss = [], [], 0.0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval test", ncols=80):
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            out    = model(**batch, labels=labels)
            total_loss += out['loss'].item()
            probs = torch.softmax(out['logits'], dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs > THRESHOLD).astype(int)
    return {
        'AUC':  roc_auc_score(all_labels, all_probs),
        'F1':   f1_score(all_labels, preds, zero_division=0, average='macro'),
        'P':    precision_score(all_labels, preds, zero_division=0, average='macro'),
        'R':    recall_score(all_labels, preds, zero_division=0, average='macro'),
        'Acc':  (preds == all_labels).mean(),
        'Loss': total_loss / len(loader),
    }


if __name__ == "__main__":
    data      = load_data(DATA_PATH)
    feat_cols = [c for c in data.columns if c not in NON_FEATURE_COLS]
    assert len(feat_cols) == 88, f"Expect 88 POIs, we found {len(feat_cols)} POIs"

    texts     = data['Text'].tolist()
    labels    = data['Depression'].tolist()
    raw_feats = data[feat_cols].values.astype(np.float32)

    train_idx, val_idx, test_idx = split_by_sample(data)

    X_train = [texts[i] for i in train_idx];  X_val = [texts[i] for i in val_idx];  X_test = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]; y_val = [labels[i] for i in val_idx]; y_test = [labels[i] for i in test_idx]
    F_train = raw_feats[train_idx]; F_val = raw_feats[val_idx]; F_test = raw_feats[test_idx]

    feat_dim = F_train.shape[1]
    print(f"--> Feature dim: {feat_dim}")

    data.iloc[test_idx].reset_index(drop=True).to_csv(TEST_CSV, index=False, encoding='utf-8-sig')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    lengths   = [len(tokenizer.encode(str(t))) for t in random.sample(texts, min(5000, len(texts)))]
    max_len   = min(int(np.percentile(lengths, 95)), 256)
    print(f"--> 95% text length: {max_len}")

    train_ds = LooMDataset(X_train, y_train, F_train, tokenizer, max_len)
    val_ds   = LooMDataset(X_val,   y_val,   F_val,   tokenizer, max_len)
    test_ds  = LooMDataset(X_test,  y_test,  F_test,  tokenizer, max_len)

    model = DualStream(MODEL_NAME, 2, feat_dim, NUM_HEADS).to(device)

    args = TrainingArguments(
        output_dir=FILEPATH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='roc_auc',
        greater_is_better=True,
        group_by_length=True,
        logging_steps=50,
        save_total_limit=2,
        seed=SEED,
        fp16=True,
        learning_rate=LEARNING_RATE,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    trainer.train()

    metrics = evaluate_with_progress(trainer.model, test_ds, tokenizer)
    result_line = (f"AUC={metrics['AUC']:.4f} | F1={metrics['F1']:.4f} | "
                   f"P={metrics['P']:.4f} | R={metrics['R']:.4f} | "
                   f"Acc={metrics['Acc']:.4f} | Loss={metrics['Loss']:.4f}")
    print(f"\n{result_line}")

    with open(RESULT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")
        f.write(result_line + "\n")

    torch.save(trainer.model.state_dict(), MODEL_SAVE_PATH)
    for ckpt in glob.glob(os.path.join(FILEPATH, "checkpoint-*")):
        if os.path.isdir(ckpt):
            shutil.rmtree(ckpt)
    torch.cuda.empty_cache()
    print(f"\nModel saved: {MODEL_SAVE_PATH}")