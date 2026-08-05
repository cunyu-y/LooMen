import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy import stats

# =============================================
INPUT_CSV        = '../../Data/City/City1_abs.csv'
POP_CSV          = '../../Data/City/population.csv'
VALID_CITIES_CSV = '../../Data/City/City_name.csv'

OUTPUT_DIR  = '../../Results/Figure4'
OUTPUT_NAME = 'forest_plot.png'

POI_CONFIGS = {
    'Education': '#624296',
    'Shopping': '#7456a5',
    'Sports': '#8e77b5',
    'Work': '#a999c5',
    'Dining': '#c4bcd4',
    'Living': '#e0dfe2',
    'Medical': '#dddfde',
    'Transportation': '#b6c7be',
    'Culture': '#90ae9f',
    'Leisure': '#6b9781',
    'Residential': '#457f63',
    'Public Space': '#317052'
}

N_BOOTSTRAP      = 1000
CONFIDENCE_LEVEL = 0.95
FIG_SIZE         = (4, 5)
# ============================

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


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

    output_columns = ['City', 'City_name', 'Population', 'Sample_Count']
    for feature in features:
        final_data[feature] = final_data[f"Avg_IG_Value_{feature}"]
        output_columns.append(feature)

    print(f"Valid cities: {len(final_data)}, Features: {len(features)}")
    return final_data[output_columns]


def fit_scaling_law(x, y):
    mask = (x > 0) & (y > 0) & ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 10:
        return None, None

    log_x, log_y = np.log10(x_clean), np.log10(y_clean)
    result = stats.linregress(log_x, log_y)
    return result.slope, result.rvalue**2


def bootstrap_ci(x, y, n_bootstrap=1000):
    mask = (x > 0) & (y > 0) & ~np.isnan(x) & ~np.isnan(y)
    x_c, y_c = x[mask], y[mask]
    n = len(x_c)
    betas = []

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            lx = np.log10(x_c[idx])
            ly = np.log10(y_c[idx])
            if len(lx) > 2:
                beta = stats.linregress(lx, ly).slope
                betas.append(beta)
        except:
            continue

    alpha = (1 - CONFIDENCE_LEVEL) / 2
    ci = np.percentile(betas, [alpha * 100, (1 - alpha) * 100])
    return ci[0], ci[1]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_attribution()

    results = []
    pop = df['Population'].values
    valid_pois = [p for p in POI_CONFIGS.keys() if p in df.columns]

    if not valid_pois:
        print("Warning")

    for poi_name in valid_pois:
        color = POI_CONFIGS[poi_name]
        score = df[poi_name].values
        total_val = score * pop

        beta, r2 = fit_scaling_law(pop, total_val)

        if beta is not None:
            ci_lower, ci_upper = bootstrap_ci(pop, total_val, N_BOOTSTRAP)
            results.append({
                'POI': poi_name,
                'Beta': beta,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'R2': r2,
                'Color': color
            })

    results_df = pd.DataFrame(results).sort_values('Beta', ascending=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    y_positions = np.arange(len(results_df))

    for i, row in results_df.iterrows():
        y_pos = y_positions[list(results_df.index).index(i)]

        ax.plot([row['CI_lower'], row['CI_upper']], [y_pos, y_pos],
                color=row['Color'], lw=2, alpha=0.6)

        ax.scatter(row['Beta'], y_pos, s=80, color=row['Color'],
                   edgecolors='k', linewidths=0.5, zorder=3)

    ax.axvline(1, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(results_df['POI'])
    ax.set_xlabel('Scaling Exponent (β)', fontsize=11)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    ax.set_ylabel('')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, ls=':')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"saved: {out_path}")
    print(results_df[['POI', 'Beta', 'CI_lower', 'CI_upper', 'R2']].to_string(index=False))


if __name__ == "__main__":
    main()