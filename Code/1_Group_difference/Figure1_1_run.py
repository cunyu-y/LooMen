import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "../../Data/Figure1/Analysis_1.csv"
TRANSLATION_FILE = "../../Data/Translation.csv"
OUTPUT_DIR = "../../Data/Figure1/"

N_BOOTSTRAP = 1000        # number of bootstrap replicates
RANDOM_SEED = 42          # fixed seed
N_JOBS = -1               # parallel workers

NEG_THRESHOLD = -0.5      # sentiment <= threshold -> strongly negative
POS_THRESHOLD = 0.5       # sentiment >= threshold -> strongly positive

# Group names follow the depression label: index 0 -> non-depressive, index 1 -> depressive
GROUPS = ['Non Depressive', 'Depressive']
SENTIMENT_CATEGORIES = ['Strongly negative', 'Moderate', 'Strongly positive']

os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(fn):
    return OUTPUT_DIR + fn


def load_translation(fp):
    table = pd.read_csv(fp, header=None, names=['source', 'english'], encoding='utf-8-sig')
    table['source'] = table['source'].astype(str).str.strip()
    table['english'] = table['english'].astype(str).str.strip()
    duplicated = sorted(set(table.loc[table['source'].duplicated(), 'source']))
    if duplicated:
        raise ValueError(f"Duplicated entries in {fp}: {duplicated}")
    return dict(zip(table['source'], table['english']))


def load_data(fp, translation):
    chunks = []
    for c in pd.read_csv(fp, chunksize=10000):
        chunks.append(c)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    df.columns = ['text', 'sentiment', 'depression', 'poi_type']

    df['poi_type'] = df['poi_type'].astype(str).str.strip()
    missing = sorted(set(df['poi_type']) - set(translation))
    if missing:
        raise KeyError(f"POI types absent from {TRANSLATION_FILE}: {missing}")
    df['poi_type'] = df['poi_type'].map(translation)
    return df


def generate_bootstrap_indices(df, n):
    is_dep = df['depression'].values == 1
    dep_idx = np.where(is_dep)[0]
    con_idx = np.where(~is_dep)[0]
    n_dep, n_con = len(dep_idx), len(con_idx)
    print(f"Sample sizes: depressive {n_dep}, non-depressive {n_con}")
    np.random.seed(RANDOM_SEED)
    indices = []
    if n_con >= n_dep:
        replace_con = n_con < n_dep * n
        sampled_con = np.random.choice(con_idx, size=(n, n_dep), replace=replace_con)
        for i in range(n):
            sampled_dep = np.random.choice(dep_idx, size=n_dep, replace=True)
            indices.append(np.concatenate([sampled_dep, sampled_con[i]]))
    else:
        for i in range(n):
            sampled_dep = np.random.choice(dep_idx, size=n_con, replace=True)
            sampled_con = np.random.choice(con_idx, size=n_con, replace=True)
            indices.append(np.concatenate([sampled_dep, sampled_con]))
    return indices


def encode_types(df):
    all_types = df['poi_type'].unique()
    t2i = {t: i for i, t in enumerate(all_types)}
    tidx = np.array([t2i[t] for t in df['poi_type'].values], dtype=np.int32)
    return all_types, tidx


def summarize(arr, axis=0):
    return np.mean(arr, axis=axis), np.percentile(arr, [2.5, 97.5], axis=axis)


def cramers_v(chi2, n, r, c):
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))


def _coverage_50(counts):
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0
    sc = np.sort(counts)[::-1]
    cum = np.cumsum(sc) / sc.sum()
    idx = np.argmax(cum >= 0.5)
    if idx == 0:
        return np.interp(0.5, [0, cum[0]], [0, 1])
    return np.interp(0.5, [cum[idx - 1], cum[idx]], [idx, idx + 1])


def _tent(vals):
    if len(vals) <= 1:
        return 0.0
    hist, _ = np.histogram(vals, bins=10, range=(-1, 1))
    hist = hist[hist > 0]
    return stats.entropy(hist, base=2) if len(hist) > 1 else 0.0


def _make_chunks(lst, n_jobs):
    k = cpu_count() if n_jobs == -1 else abs(n_jobs)
    segs = np.array_split(np.arange(len(lst)), min(k, len(lst)))
    return [[lst[i] for i in seg] for seg in segs]


def _w_pattern(chunk, scores, dep_labels, neg_thr, pos_thr):
    n = len(chunk)
    chi2_v = np.zeros(n)
    pv = np.ones(n)
    cv_v = np.zeros(n)
    pct = np.zeros((n, 6))

    for i, idx in enumerate(chunk):
        s = scores[idx]
        dl = dep_labels[idx]
        cats = np.select([s <= neg_thr, s >= pos_thr], [0, 2], default=1)
        ct = np.zeros((2, 3), dtype=int)
        for g in range(2):
            m = dl == g
            if m.any():
                ct[g] = np.bincount(cats[m], minlength=3)
        rs = ct.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        pct[i] = (ct / rs * 100).flatten()
        if ct.sum() > 0 and (ct.sum(axis=0) > 0).all() and (ct.sum(axis=1) > 0).all():
            chi2, p, _, _ = stats.chi2_contingency(ct)
            chi2_v[i] = chi2
            pv[i] = p
            cv_v[i] = cramers_v(chi2, ct.sum(), 2, 3)

    return chi2_v, pv, cv_v, pct


def _w_breadth(chunk, scores, dep_labels, tidx, n_types):
    n = len(chunk)
    d_cov = np.zeros(n)
    c_cov = np.zeros(n)
    d_prop = np.zeros((n, n_types))
    c_prop = np.zeros((n, n_types))
    o_sent = np.zeros((n, n_types))
    d_tent = np.zeros((n, n_types))
    c_tent = np.zeros((n, n_types))

    for i, idx in enumerate(chunk):
        s_dep = dep_labels[idx] == 1
        d_ti = tidx[idx][s_dep]
        c_ti = tidx[idx][~s_dep]
        d_s = scores[idx][s_dep]
        c_s = scores[idx][~s_dep]
        s_ti = tidx[idx]

        if len(d_ti):
            _, c = np.unique(d_ti, return_counts=True)
            d_cov[i] = _coverage_50(c)
        if len(c_ti):
            _, c = np.unique(c_ti, return_counts=True)
            c_cov[i] = _coverage_50(c)

        nd = len(d_ti)
        nc = len(c_ti)
        for j in range(n_types):
            om = s_ti == j
            if om.any():
                o_sent[i, j] = np.mean(scores[idx][om])
            dm = d_ti == j
            d_prop[i, j] = dm.sum() / nd if nd else 0
            if dm.any():
                d_tent[i, j] = _tent(d_s[dm])
            cm = c_ti == j
            c_prop[i, j] = cm.sum() / nc if nc else 0
            if cm.any():
                c_tent[i, j] = _tent(c_s[cm])

    return d_cov, c_cov, d_prop, c_prop, o_sent, d_tent, c_tent


def _w_poi_entropy(chunk, scores, dep_labels, tidx, neg_thr, pos_thr):
    n = len(chunk)
    ent = np.zeros((n, 2, 3))

    for i, idx in enumerate(chunk):
        s_ti = tidx[idx]
        dl = dep_labels[idx]
        s = scores[idx]
        cats = np.select([s <= neg_thr, s >= pos_thr], [0, 2], default=1)
        for g in range(2):
            for k in range(3):
                mask = (dl == g) & (cats == k)
                if mask.any():
                    _, c = np.unique(s_ti[mask], return_counts=True)
                    ent[i, g, k] = stats.entropy(c, base=2) if len(c) > 1 else 0

    return ent


def analyze_expression_pattern(df, boot_idx):
    print("\n[Analysis 1] Sentiment expression pattern")
    scores = df['sentiment'].values
    dep_labels = df['depression'].values

    rs = Parallel(n_jobs=N_JOBS)(
        delayed(_w_pattern)(c, scores, dep_labels, NEG_THRESHOLD, POS_THRESHOLD)
        for c in _make_chunks(boot_idx, N_JOBS)
    )

    chi2_v = np.concatenate([r[0] for r in rs])
    pv = np.concatenate([r[1] for r in rs])
    cv_v = np.concatenate([r[2] for r in rs])
    pct = np.concatenate([r[3] for r in rs])

    chi2_m, chi2_ci = summarize(chi2_v)
    cv_m, cv_ci = summarize(cv_v)
    p_m, p_ci = summarize(pv)
    pct_m, _ = summarize(pct)

    print(f"  Chi-square:  {chi2_m:.4f}  95% CI [{chi2_ci[0]:.4f}, {chi2_ci[1]:.4f}]")
    print(f"  Cramer's V:  {cv_m:.4f}  95% CI [{cv_ci[0]:.4f}, {cv_ci[1]:.4f}]")
    print(f"  P value:     {p_m:.3e}")

    pd.DataFrame({
        'Statistic': ['Chi-square', 'Cramers V', 'P value'],
        'Mean': [chi2_m, cv_m, p_m],
        'CI lower': [chi2_ci[0], cv_ci[0], p_ci[0]],
        'CI upper': [chi2_ci[1], cv_ci[1], p_ci[1]],
    }).to_csv(out("sentiment_pattern_statistics.csv"), index=False, encoding='utf-8-sig')

    pd.DataFrame({
        'Sentiment pattern': [f"{g}_{c}" for g in GROUPS for c in SENTIMENT_CATEGORIES],
        'Mean percentage': pct_m,
    }).to_csv(out("sentiment_pattern_percentage.csv"), index=False, encoding='utf-8-sig')


def analyze_topic_breadth(df, boot_idx, all_types, tidx):
    print("\n[Analysis 2] Topic breadth")
    scores = df['sentiment'].values
    dep_labels = df['depression'].values

    rs = Parallel(n_jobs=N_JOBS)(
        delayed(_w_breadth)(c, scores, dep_labels, tidx, len(all_types))
        for c in _make_chunks(boot_idx, N_JOBS)
    )

    d_cov = np.concatenate([r[0] for r in rs])
    c_cov = np.concatenate([r[1] for r in rs])
    d_prop = np.concatenate([r[2] for r in rs])
    c_prop = np.concatenate([r[3] for r in rs])
    o_sent = np.concatenate([r[4] for r in rs])
    d_tent = np.concatenate([r[5] for r in rs])
    c_tent = np.concatenate([r[6] for r in rs])

    d_cov_m, d_cov_ci = summarize(d_cov)
    c_cov_m, c_cov_ci = summarize(c_cov)

    print(f"  Coverage depressive:      {d_cov_m:.2f}  95% CI [{d_cov_ci[0]:.2f}, {d_cov_ci[1]:.2f}]")
    print(f"  Coverage non-depressive:  {c_cov_m:.2f}  95% CI [{c_cov_ci[0]:.2f}, {c_cov_ci[1]:.2f}]")

    pd.DataFrame({
        'Metric': ['Coverage_Depressive', 'Coverage_Non Depressive'],
        'Mean': [d_cov_m, c_cov_m],
        'CI lower': [d_cov_ci[0], c_cov_ci[0]],
        'CI upper': [d_cov_ci[1], c_cov_ci[1]],
    }).to_csv(out("topic_breadth_summary.csv"), index=False, encoding='utf-8-sig')

    d_pm, d_pci = summarize(d_prop)
    c_pm, c_pci = summarize(c_prop)
    o_sm, _ = summarize(o_sent)
    d_tem, _ = summarize(d_tent)
    c_tem, _ = summarize(c_tent)

    pd.DataFrame({
        'POI type': all_types,
        'Depressive proportion': d_pm,
        'Depressive proportion CI lower': d_pci[0],
        'Depressive proportion CI upper': d_pci[1],
        'Non Depressive proportion': c_pm,
        'Non Depressive proportion CI lower': c_pci[0],
        'Non Depressive proportion CI upper': c_pci[1],
        'Overall sentiment mean': o_sm,
        'Depressive within-type entropy': d_tem,
        'Non Depressive within-type entropy': c_tem,
    }).sort_values(['Depressive proportion', 'Non Depressive proportion'],
                   ascending=False).to_csv(
        out("topic_breadth_by_type.csv"), index=False, encoding='utf-8-sig')


def analyze_poi_entropy(df, boot_idx, tidx):
    print("\n[Analysis 3] POI entropy by sentiment category")
    scores = df['sentiment'].values
    dep_labels = df['depression'].values

    rs = Parallel(n_jobs=N_JOBS)(
        delayed(_w_poi_entropy)(c, scores, dep_labels, tidx, NEG_THRESHOLD, POS_THRESHOLD)
        for c in _make_chunks(boot_idx, N_JOBS)
    )

    ent = np.concatenate(rs)
    ent_m, ent_ci = summarize(ent)

    for g in range(2):
        for s in range(3):
            print(f"  {GROUPS[g]} | {SENTIMENT_CATEGORIES[s]}:  H = {ent_m[g, s]:.4f}  "
                  f"95% CI [{ent_ci[0, g, s]:.4f}, {ent_ci[1, g, s]:.4f}]")

    pd.DataFrame([
        {'Group': GROUPS[g],
         'Sentiment category': SENTIMENT_CATEGORIES[s],
         'POI entropy mean': ent_m[g, s],
         'POI entropy CI lower': ent_ci[0, g, s],
         'POI entropy CI upper': ent_ci[1, g, s]}
        for g in range(2) for s in range(3)
    ]).to_csv(out("poi_entropy_by_sentiment.csv"), index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    t0 = time.time()

    translation = load_translation(TRANSLATION_FILE)
    df = load_data(INPUT_FILE, translation)
    print(f"Loaded {len(df)} records, {df['poi_type'].nunique()} POI types")

    boot_idx = generate_bootstrap_indices(df, N_BOOTSTRAP)
    all_types, tidx = encode_types(df)

    analyze_expression_pattern(df, boot_idx)
    analyze_topic_breadth(df, boot_idx, all_types, tidx)
    analyze_poi_entropy(df, boot_idx, tidx)

    del df
    gc.collect()

    elapsed = int(time.time() - t0)
    print(f"\nDone in {elapsed // 60} min {elapsed % 60} s")