import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ===================================================
INPUT_CSV        = '../../Data/City/City2_abs.csv'
POP_CSV          = '../../Data/City/population.csv'
VALID_CITIES_CSV = '../../Data/City/City_name.csv'

OUTPUT_DIR  = '../../Results/Figure4'
OUTPUT_NAME = 'Scaling_{feature}.png'

PLOT_FEATURES = ['Cinema', 'Resort']

POI_COLORS = {
    'Resort': '#B33B48', 'Temple': '#7456A5', 'Lake': '#2057A7', 'Museum': '#4A7B9D',
    'Scenic Area': '#218F8D', 'Internet Cafe': '#457F63', 'Green Space': '#D9953E', 'Cinema': '#A44269',
    'University': '#7fa3d3', 'Tea House': '#96beab', 'Office Building': '#afa0d0', 'Food Street': '#db8f98',
    'Supermarket': '#ecc991',
}
# ============================

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'

BASE_COLS = ['City', 'City_name', 'Population', 'Sample_Count']


def normalize_name(name):
    if isinstance(name, str):
        return name.replace('市', '').replace('地区', '').replace('自治州', '').replace('盟', '').strip()
    return name


def load_attribution():
    pop_data = pd.read_csv(POP_CSV)
    pop_data['City_name'] = pop_data['name'].apply(normalize_name)
    pop_data['Population'] = pd.to_numeric(pop_data['人口总数_平均'], errors='coerce')
    valid_cities_set = set(pd.read_csv(VALID_CITIES_CSV)['City'])

    csv_data = pd.read_csv(INPUT_CSV)
    features = [col.replace("Avg_IG_Value_", "") for col in csv_data.columns if col.startswith("Avg_IG_Value_")]

    csv_data['City_name'] = csv_data['City'].apply(normalize_name)
    merged = csv_data.merge(pop_data[['City_name', 'Population']], on='City_name')
    final_data = merged[merged['City'].isin(valid_cities_set)].copy()

    output_columns = BASE_COLS[:]
    for feature in features:
        final_data[feature] = final_data[f"Avg_IG_Value_{feature}"]
        output_columns.append(feature)

    print(f"Valid cities: {len(final_data)}, Features: {len(features)}")
    return final_data[output_columns]


def get_bootstrap_ci(x, y, n_boot=1000):
    if len(x) < 3:
        return [0, 0]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), (n_boot, len(x)), replace=True)
    lx, ly = np.log10(x), np.log10(y)
    x_mean = np.mean(lx[idx], axis=1, keepdims=True)
    y_mean = np.mean(ly[idx], axis=1, keepdims=True)
    num = np.sum((lx[idx] - x_mean) * (ly[idx] - y_mean), axis=1)
    den = np.sum((lx[idx] - x_mean) ** 2, axis=1)
    return np.percentile(num / den, [2.5, 97.5])


def calc_adj_r2(r_squared, n, p=1):
    if n <= p + 1:
        return 0
    return 1 - (1 - r_squared) * (n - 1) / (n - p - 1)


def plot_scaling(df, poi_colors, out_dir):
    pop_all = df['Population'].values
    cache = []

    for feat in poi_colors:
        if feat not in df.columns:
            continue
        score = df[feat].values
        total = score * pop_all
        mask = (pop_all > 0) & (total > 0) & (score > 0)
        x = pop_all[mask]
        if len(x) < 5:
            continue
        cache.append({
            'feat': feat,
            'x': x,
            'y_total': total[mask],
            'y_mean': score[mask],
            'res_t': stats.linregress(np.log10(x), np.log10(total[mask])),
            'res_m': stats.linregress(np.log10(x), np.log10(score[mask])),
            'n': len(x),
        })

    if not cache:
        return

    _, p_adj_t, _, _ = multipletests([item['res_t'].pvalue for item in cache], method='fdr_bh')
    _, p_adj_m, _, _ = multipletests([item['res_m'].pvalue for item in cache], method='fdr_bh')

    for i, item in enumerate(cache):
        feat = item['feat']
        if feat not in PLOT_FEATURES:
            continue
        x = item['x']
        n = item['n']
        color = poi_colors[feat]
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)

        rt, rm = item['res_t'], item['res_m']
        ci_t = get_bootstrap_ci(x, item['y_total'])
        ci_m = get_bootstrap_ci(x, item['y_mean'])
        r2_adj_t = calc_adj_r2(rt.rvalue ** 2, n)
        r2_adj_m = calc_adj_r2(rm.rvalue ** 2, n)
        p_str_t = "<0.001" if p_adj_t[i] < 0.001 else f"={p_adj_t[i]:.3f}"
        p_str_m = "<0.001" if p_adj_m[i] < 0.001 else f"={p_adj_m[i]:.3f}"

        fig = plt.figure(figsize=(6, 3))
        ax1 = fig.add_axes([0.10, 0.18, 0.38, 0.70])
        ax2 = fig.add_axes([0.58, 0.18, 0.38, 0.70])

        for ax, y, res, ci, r2_adj, p_str, ylabel, title_prefix in [
            (ax1, item['y_total'], rt, ci_t, r2_adj_t, p_str_t, 'Total Attribution', 'Total'),
            (ax2, item['y_mean'], rm, ci_m, r2_adj_m, p_str_m, 'Average Attribution', 'Per Capita'),
        ]:
            ax.scatter(x, y, s=25, alpha=0.3, c=color, edgecolors='none')
            ax.scatter(x, y, s=25, alpha=0.7, c='none', edgecolors='k', linewidths=0.2)
            ax.plot(x_fit, (10 ** res.intercept) * (x_fit ** res.slope), 'k-', lw=1.5, alpha=0.8)
            ax.set_title(
                f"{title_prefix}: {feat}; β={res.slope:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]\n"
                f"R²={r2_adj:.2f}, p{p_str}", fontsize=8)
            ax.set_xlabel('Population')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.text(0.05, 0.95, f'N={n}', transform=ax.transAxes, va='top', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        out_path = f"{out_dir}/{OUTPUT_NAME.format(feature=feat)}"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_attribution()
    plot_scaling(df, POI_COLORS, OUTPUT_DIR)
    print("Done")


if __name__ == "__main__":
    main()