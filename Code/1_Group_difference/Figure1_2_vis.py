import os
import warnings

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

DATA_DIR = "../../Data/Figure1"
FIG_DIR = "../../Results/Figure1"

INPUT_BREADTH_SUMMARY = f"{DATA_DIR}/topic_breadth_summary.csv"
INPUT_BREADTH_TYPES = f"{DATA_DIR}/topic_breadth_by_type.csv"
INPUT_PATTERN_PCT = f"{DATA_DIR}/sentiment_pattern_percentage.csv"
INPUT_PATTERN_STATS = f"{DATA_DIR}/sentiment_pattern_statistics.csv"
INPUT_POI_ENTROPY = f"{DATA_DIR}/poi_entropy_by_sentiment.csv"

OUTPUT_1A = f"{FIG_DIR}/Fig1a.png"
OUTPUT_1B = f"{FIG_DIR}/Fig1b.png"
OUTPUT_1C = f"{FIG_DIR}/Fig1c.png"

# Group colours, shared by Fig.1a and Fig.1c
COLOR_DEP = '#7456A5'      # depressive group -> purple
COLOR_CON = '#457F63'      # non-depressive group -> green

# Sentiment colours for Fig.1b, aligned with the group colours
COLOR_NEG = '#7456A5'      # strongly negative -> purple
COLOR_NEU = '#6D7E92'      # moderate / neutral -> grey
COLOR_POS = '#457F63'      # strongly positive -> green

# Row labels written by the analysis script
GROUPS = [('Non Depressive', 'Control'), ('Depressive', 'Depression')]
SENTIMENT_CATEGORIES = ['Strongly negative', 'Moderate', 'Strongly positive']
SENTIMENT_LEGEND = ['Strongly negative', 'Moderate', 'Strongly positive']
SENTIMENT_COLORS = [COLOR_NEG, COLOR_NEU, COLOR_POS]

# Fig.1a: maximum number of ranked POI types shown on the cumulative curve
MAX_POI_SHOW = 40

# Fig.1c: POI types that get an on-plot label; names must match Translation.csv
LABEL_TYPES = [
    'Green Space', 'Hospital', 'Dessert Shop', 'Scenic Area', 'Market',
    'Art Gallery', 'Internet Cafe', 'Cafe', 'Pharmacy', 'Food Street',
    'Resort', 'Emergency Center', 'University', 'Gym', 'Subway Station',
    'Airport', 'Library',
]

os.makedirs(FIG_DIR, exist_ok=True)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.frameon': False,
    'axes.grid': False,
})

# ============================================================
# Helpers
# ============================================================


def integer_percentages(values):
    values = np.asarray(values, dtype=float)
    floors = np.floor(values).astype(int)
    shortfall = int(round(values.sum())) - int(floors.sum())
    for i in np.argsort(-(values - floors))[:max(shortfall, 0)]:
        floors[i] += 1
    return floors


def format_p(p):
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"


def spearman_label(y, x):
    if len(x) <= 3:
        return "N/A"
    r, p = stats.spearmanr(y, x)
    return f"\u03c1 = {r:.3f}, {format_p(p)}"


def setup_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('outward', 0))


def load_scatter_data():
    df = pd.read_csv(INPUT_BREADTH_TYPES, encoding='utf-8-sig')

    unknown = sorted(set(LABEL_TYPES) - set(df['POI type']))
    if unknown:
        print(f"  Warning: LABEL_TYPES not found in the data: {unknown}")

    sentiment = df['Overall sentiment mean'].values
    dep_ent = df['Depressive within-type entropy'].values
    con_ent = df['Non Depressive within-type entropy'].values
    types = df['POI type'].values

    dep_valid = (dep_ent > 1e-6) & (np.abs(sentiment) > 1e-6)
    con_valid = (con_ent > 1e-6) & (np.abs(sentiment) > 1e-6)

    return (
        {'x': sentiment[dep_valid], 'y': dep_ent[dep_valid], 'types': types[dep_valid]},
        {'x': sentiment[con_valid], 'y': con_ent[con_valid], 'types': types[con_valid]},
    )


# ============================================================
# Fig.1a  Cumulative POI coverage curve
# ============================================================


def plot_coverage():
    summary = pd.read_csv(INPUT_BREADTH_SUMMARY, encoding='utf-8-sig').set_index('Metric')
    types_df = pd.read_csv(INPUT_BREADTH_TYPES, encoding='utf-8-sig')

    dep_cov = summary.loc['Coverage_Depressive', 'Mean']
    con_cov = summary.loc['Coverage_Non Depressive', 'Mean']

    dep_sorted = types_df.sort_values('Depressive proportion', ascending=False).reset_index(drop=True)
    con_sorted = types_df.sort_values('Non Depressive proportion', ascending=False).reset_index(drop=True)

    n = min(MAX_POI_SHOW, len(types_df))
    x_pos = np.arange(1, n + 1)
    dep_cum = np.cumsum(dep_sorted['Depressive proportion'].values)
    con_cum = np.cumsum(con_sorted['Non Depressive proportion'].values)

    fig, ax = plt.subplots(figsize=(2.8, 2.4))

    ax.fill_between(x_pos,
                    np.cumsum(dep_sorted['Depressive proportion CI lower'].values[:n]) * 100,
                    np.cumsum(dep_sorted['Depressive proportion CI upper'].values[:n]) * 100,
                    alpha=0.2, color=COLOR_DEP, linewidth=0)
    ax.fill_between(x_pos,
                    np.cumsum(con_sorted['Non Depressive proportion CI lower'].values[:n]) * 100,
                    np.cumsum(con_sorted['Non Depressive proportion CI upper'].values[:n]) * 100,
                    alpha=0.2, color=COLOR_CON, linewidth=0)

    ax.plot(x_pos, dep_cum[:n] * 100, color=COLOR_DEP, linewidth=1.5, label='Depressive')
    ax.plot(x_pos, con_cum[:n] * 100, color=COLOR_CON, linewidth=1.5, label='Non-Depressive')

    ax.axhline(50, color='k', linestyle='--', alpha=0.9, linewidth=0.2)
    ax.axvline(dep_cov, color=COLOR_DEP, linestyle=':', alpha=0.8, linewidth=1)
    ax.axvline(con_cov, color=COLOR_CON, linestyle=':', alpha=0.8, linewidth=1)
    ax.text(dep_cov + 0.5, 45, f'{dep_cov:.1f}', color='k', fontsize=7, ha='left', va='center')
    ax.text(con_cov + 0.5, 55, f'{con_cov:.1f}', color='k', fontsize=7, ha='left', va='center')

    ax.set_xlabel('POI rank', fontsize=8)
    ax.set_ylabel('Cumulative proportion (%)', fontsize=8)
    ax.set_xlim(1, n)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=7)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_1A, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_1A}")


# ============================================================
# Fig.1b  Sentiment expression pattern
# ============================================================


def plot_sentiment_pattern():
    pct = pd.read_csv(INPUT_PATTERN_PCT, encoding='utf-8-sig').set_index('Sentiment pattern')
    stat = pd.read_csv(INPUT_PATTERN_STATS, encoding='utf-8-sig').set_index('Statistic')
    poi_ent = pd.read_csv(INPUT_POI_ENTROPY, encoding='utf-8-sig').set_index(
        ['Group', 'Sentiment category'])

    values = np.array([[pct.loc[f"{key}_{cat}", 'Mean percentage'] for cat in SENTIMENT_CATEGORIES]
                       for key, _ in GROUPS])
    entropy = np.array([[poi_ent.loc[(key, cat), 'POI entropy mean'] for cat in SENTIMENT_CATEGORIES]
                        for key, _ in GROUPS])
    shown = np.array([integer_percentages(row) for row in values])

    header = (f"Chi-square test: {format_p(stat.loc['P value', 'Mean'])}, "
              f"Cramer's V = {stat.loc['Cramers V', 'Mean']:.3f} "
              f"[{stat.loc['Cramers V', 'CI lower']:.3f}, {stat.loc['Cramers V', 'CI upper']:.3f}]")

    y_pos = np.array([0.3, 0.7])
    fig, ax = plt.subplots(figsize=(3.6, 2.4))

    left = np.zeros(len(GROUPS))
    for k, color in enumerate(SENTIMENT_COLORS):
        vals = values[:, k]
        ax.barh(y_pos, vals, left=left, color=color, height=0.25,
                edgecolor='black', linewidth=0.6, alpha=0.85)
        for i in range(len(GROUPS)):
            if vals[i] > 10:
                center = left[i] + vals[i] / 2
                ax.text(center, y_pos[i] + 0.048, f'{entropy[i, k]:.2f}',
                        ha='center', va='center', fontsize=6, color='white')
                ax.text(center, y_pos[i] - 0.042, f'{shown[i, k]}%',
                        ha='center', va='center', fontsize=7, fontweight='bold', color='white')
        left += vals

    ax.set_title(header, fontsize=7.5, pad=6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for _, label in GROUPS], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black',
                                  linewidth=0.5, label=lab)
               for c, lab in zip(SENTIMENT_COLORS, SENTIMENT_LEGEND)]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=7, handlelength=1.0,
              handletextpad=0.3, columnspacing=1.0)
    ax.text(0.5, -0.34, 'Upper value: POI entropy (bits)   Lower value: share of posts',
            transform=ax.transAxes, ha='center', va='center', fontsize=6, color='0.35')

    plt.tight_layout()
    plt.savefig(OUTPUT_1B, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_1B}")


# ============================================================
# Fig.1c  Within-type entropy vs sentiment score
# ============================================================


def plot_scatter(dep_data, con_data):
    dep_label = spearman_label(dep_data['y'], dep_data['x'])
    con_label = spearman_label(con_data['y'], con_data['x'])
    print(f"  Spearman Depressive: {dep_label}")
    print(f"  Spearman Non-Depressive:    {con_label}")

    fig, ax = plt.subplots(figsize=(6, 6))

    for data, color, marker, name in [
        (con_data, COLOR_CON, 's', 'Non-Depressive'),
        (dep_data, COLOR_DEP, 'o', 'Depressive'),
    ]:
        hi = np.isin(data['types'], LABEL_TYPES)
        lo = ~hi
        if lo.any():
            ax.scatter(data['y'][lo], data['x'][lo], c=color, s=80, alpha=0.2,
                       edgecolors=color, linewidth=0.5, marker=marker, label=name)
        if hi.any():
            ax.scatter(data['y'][hi], data['x'][hi], c=color, s=80, alpha=0.8,
                       edgecolors='k', linewidth=0.5, marker=marker)
            for xi, yi, t in zip(data['y'][hi], data['x'][hi], data['types'][hi]):
                ax.annotate(t, xy=(xi, yi), xytext=(xi + 0.01, yi + 0.01),
                            fontsize=8, color='k', ha='left', va='center')

    setup_axis(ax, 'Sentiment Entropy', 'Average Sentiment of POls')
    ax.legend(fontsize=9)

    fig.text(0.5, 0.92, f"Depressive: {dep_label}", ha='center', va='top', fontsize=9, color=COLOR_DEP)
    fig.text(0.5, 0.89, f"Non-Depressive: {con_label}", ha='center', va='top', fontsize=9, color=COLOR_CON)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(xlim[0] - 0.02, xlim[1] + 0.05)
    ax.set_ylim(ylim[0] - 0.05, ylim[1] + 0.15)
    plt.subplots_adjust(top=0.85)

    plt.savefig(OUTPUT_1C, dpi=600, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_1C}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("[Fig.1a] Coverage curve")
    plot_coverage()

    print("[Fig.1b] Sentiment expression pattern")
    plot_sentiment_pattern()

    print("[Fig.1c] Entropy vs sentiment")
    plot_scatter(*load_scatter_data())

    print("\nDone.")