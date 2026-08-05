import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

# ─────────────────────────────────────────────

NPZ_FILE     = "../../Data/LooMen/Data_2_IG_results.npz"
PARQUET_FILE = "../../Data/Cluster/cluster_result.parquet"
POI_PATH     = "../../Data/POI_eng.csv"

SAVE_DIR    = "../../Results/Figure3"
OUTPUT_NAME = "Figure3a.png"
OUTPUT_DPI  = 300

TOP_CLUSTERS    = 8
CLUSTER_ORDER   = None

MAIN_CATEGORIES = ['Dining', 'Culture', 'Living', 'Transportation', 'Medical', 'Residential', 'Leisure', 'Education']
COLORS          = "#7fa3d3,#96beab,#afa0d0,#db8f98,#ecc991,#81cfce,#d095ac,#9abcd6"

FONT_SIZE       = 10
LEGEND_FONTSIZE = 8

BAR_ALPHA       = 0.7
BAR_EDGEWIDTH   = 0.5

HATCH_DENSITY   = 10
HATCH_LINEWIDTH = 0.2
HATCH_ALPHA     = 0.5
HATCH_COLOR     = 'black'

TOP2_FONTSIZE   = 7.5
SHOW_TOP2_LABEL = False
Y_PAD_TOP       = 0.15
Y_PAD_BOTTOM    = 0.05

LEGEND_NCOL     = 12
# ─────────────────────────────────────────────

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = FONT_SIZE
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


def load_data():
    data = np.load(NPZ_FILE, mmap_mode='r', allow_pickle=True)
    ig = np.array(data['feature_ig_array'] if 'feature_ig_array' in data else data['shap_values'], dtype=np.float32)
    feature_names = data['feature_names'].tolist()
    df = pl.read_parquet(PARQUET_FILE)
    cluster_labels = df['cluster'].to_numpy()
    poi_df = pd.read_csv(POI_PATH, encoding='utf-8')
    poi_to_category = {}
    for _, row in poi_df.iterrows():
        for poi in row['Secondary Category'].split('、'):
            poi_to_category[poi.strip()] = row['Primary Category']
    return ig, feature_names, cluster_labels, poi_to_category


def build_attribution_matrix(ig, feature_names, cluster_labels, poi_to_category):
    valid_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])
    total = (cluster_labels != -1).sum()
    records = []
    for cl in valid_clusters:
        mask = cluster_labels == cl
        mean_ig = ig[mask].mean(axis=0)
        row = {'cluster': str(cl), 'size_pct': float(mask.sum()) / total * 100}
        for fname, val in zip(feature_names, mean_ig):
            row[fname] = val
        records.append(row)
    return pd.DataFrame(records)


def aggregate_by_category(attr_df, feature_names, poi_to_category):
    all_categories = MAIN_CATEGORIES + ['Others']
    color_list = [c.strip() for c in COLORS.split(',')]
    color_map = dict(zip(MAIN_CATEGORIES, color_list))
    color_map['Others'] = '#E5E5E5'
    result = {'cluster': attr_df['cluster'].tolist(), 'size_pct': attr_df['size_pct'].tolist()}
    for cat in all_categories:
        result[cat] = []
    for _, row in attr_df.iterrows():
        cat_sums = {cat: 0.0 for cat in all_categories}
        for fname in feature_names:
            if fname not in row:
                continue
            cat = poi_to_category.get(fname, 'Unknown')
            key = cat if cat in MAIN_CATEGORIES else 'Others'
            cat_sums[key] += row[fname]
        for cat in all_categories:
            result[cat].append(cat_sums[cat])
    return pd.DataFrame(result), color_map


def draw_bar_separated(ax, x, heights, bottoms, width, color, edgecolor, edgewidth, alpha, label):
    ax.bar(x, heights, width, bottom=bottoms,
           color=color, alpha=alpha, edgecolor='none', linewidth=0, label=label)
    ax.bar(x, heights, width, bottom=bottoms,
           color='none', alpha=1.0, edgecolor=edgecolor, linewidth=edgewidth)


def plot(agg_df, color_map):
    if CLUSTER_ORDER is None:
        agg_df = agg_df.sort_values('size_pct', ascending=False).reset_index(drop=True)
        if TOP_CLUSTERS is not None:
            agg_df = agg_df.head(TOP_CLUSTERS).reset_index(drop=True)
    else:
        agg_df = agg_df.set_index('cluster').loc[CLUSTER_ORDER].reset_index()

    clusters = agg_df['cluster'].tolist()
    size_pcts = agg_df['size_pct'].tolist()
    all_categories = MAIN_CATEGORIES + ['Others']
    active_categories = [cat for cat in all_categories if agg_df[cat].abs().sum() > 0]

    top2_per_cluster = []
    for _, row in agg_df.iterrows():
        cat_vals = {cat: abs(row[cat]) for cat in active_categories}
        top2 = sorted(cat_vals, key=cat_vals.get, reverse=True)[:2]
        top2_per_cluster.append(top2)

    rcParams['hatch.linewidth'] = HATCH_LINEWIDTH
    hatch = '/' * max(1, min(HATCH_DENSITY, 6))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
    x = np.arange(len(clusters))
    width = 0.6

    bottom_pos = np.zeros(len(clusters))
    bottom_neg = np.zeros(len(clusters))
    bar_rects = {}

    for cat in active_categories:
        values = np.array(agg_df[cat].tolist(), dtype=float)
        pos_vals = np.where(values > 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)
        color = color_map[cat]

        if pos_vals.any():
            draw_bar_separated(ax, x, pos_vals, bottom_pos, width,
                               color=color, edgecolor='k',
                               edgewidth=BAR_EDGEWIDTH, alpha=BAR_ALPHA,
                               label=cat)
            for ci in range(len(clusters)):
                if pos_vals[ci] > 0:
                    bar_rects[(ci, cat)] = (x[ci], bottom_pos[ci], pos_vals[ci])
            bottom_pos += pos_vals

        if neg_vals.any():
            draw_bar_separated(ax, x, neg_vals, bottom_neg, width,
                               color=color, edgecolor='k',
                               edgewidth=BAR_EDGEWIDTH, alpha=BAR_ALPHA,
                               label=None)
            for ci in range(len(clusters)):
                if neg_vals[ci] < 0:
                    key = (ci, cat)
                    if key not in bar_rects or abs(neg_vals[ci]) > abs(bar_rects[key][2]):
                        bar_rects[key] = (x[ci], bottom_neg[ci] + neg_vals[ci], abs(neg_vals[ci]))
            bottom_neg += neg_vals

    for ci, top2 in enumerate(top2_per_cluster):
        for cat in top2:
            key = (ci, cat)
            if key not in bar_rects:
                continue
            xc, yb, h = bar_rects[key]
            rect = mpatches.Rectangle(
                (xc - width / 2, yb), width, h,
                fill=False,
                hatch=hatch,
                edgecolor=(*plt.matplotlib.colors.to_rgb(HATCH_COLOR), HATCH_ALPHA),
                linewidth=0,
                zorder=5
            )
            ax.add_patch(rect)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'Cluster {c}\n({s:.1f}%)' for c, s in zip(clusters, size_pcts)],
        fontsize=FONT_SIZE * 0.9
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE * 0.9)
    ax.set_ylabel('Cumulative attribution value', fontsize=FONT_SIZE * 1.1)
    ax.set_xlabel('Clusters', fontsize=FONT_SIZE * 1.1)
    ax.set_title('Cumulative effects of scene categories across clusters',
                 fontsize=FONT_SIZE * 1.2, pad=12)

    if SHOW_TOP2_LABEL:
        for ci, (top2, bp) in enumerate(zip(top2_per_cluster, bottom_pos)):
            ax.text(x[ci], bp, '\n'.join(top2),
                    ha='center', va='bottom',
                    fontsize=TOP2_FONTSIZE, linespacing=1.4)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    legend = ax.legend(
        seen.values(), seen.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=LEGEND_NCOL,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    for handle in legend.legend_handles:
        handle.set_height(8)
        handle.set_width(8)
        handle.set_alpha(None)
        fc = handle.get_facecolor()
        handle.set_facecolor((*fc[:3], BAR_ALPHA))
        handle.set_edgecolor((0, 0, 0, 1.0))
        handle.set_linewidth(0.6)

    ax.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.5, color='#CCCCCC')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    y_min = min(bottom_neg.min(), 0)
    y_max = bottom_pos.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * Y_PAD_BOTTOM, y_max + y_range * Y_PAD_TOP)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    os.makedirs(SAVE_DIR, exist_ok=True)
    output_path = f"{SAVE_DIR}/{OUTPUT_NAME}"
    plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Done: {output_path}")


def main():
    ig, feature_names, cluster_labels, poi_to_category = load_data()
    attr_df = build_attribution_matrix(ig, feature_names, cluster_labels, poi_to_category)
    agg_df, color_map = aggregate_by_category(attr_df, feature_names, poi_to_category)
    plot(agg_df, color_map)


if __name__ == "__main__":
    main()