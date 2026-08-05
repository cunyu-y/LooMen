import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import umap
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================

INPUT_CSV   = '../../Data/City/City2.csv'
OUTPUT_DIR  = '../../Results/Figure4'
OUTPUT_NAME = 'UMAP.png'

MIN_SAMPLES    = 100
DEPRESSION_COL = 'Avg_Value_depression_score'
SIZE_COL       = 'Sample_Count'

UMAP_PARAMS = dict(n_components=2, metric='cosine', spread=1.0)
SEED        = 9
N_NEIGHBORS = 20
MIN_DIST    = 0.2

ROTATE_DEG = 0
FLIP       = None

ANNOTATE_TOP_N   = 15
ANNOTATE_EXTREME = 0

DOT_COLOR      = '#4E00BD'
DOT_ALPHA      = 0.15
EDGE_ALPHA     = 0.80
EDGE_ALPHA_TOP = 0.90
EDGE_WIDTH     = 0.3
EDGE_WIDTH_TOP = 0.9

FIG_SIZE = (5, 6)
DOT_SIZE = (3, 500)

LEGEND_QUANTILES = [0.15, 0.5, 0.85]
LEGEND_LABELS    = ['15th pct', 'Median', '85th pct']
# ============================================================


os.makedirs(OUTPUT_DIR, exist_ok=True)


def setup_style():
    plt.rcParams.update({
        'font.family': 'Microsoft Yahei',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 3.0,  'ytick.major.size': 3.0,
        'axes.labelsize': 9, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8,
    })


def transform_embedding(emb):
    result = emb.copy()
    if ROTATE_DEG != 0:
        rad = np.deg2rad(ROTATE_DEG)
        R = np.array([[np.cos(rad), -np.sin(rad)],
                      [np.sin(rad),  np.cos(rad)]])
        result = result @ R.T
    if FLIP and 'h' in FLIP:
        result[:, 0] = -result[:, 0]
    if FLIP and 'v' in FLIP:
        result[:, 1] = -result[:, 1]
    return result


def scale_sizes(values, power=0.8, lo=None, hi=None):
    lo = lo if lo is not None else values.min()
    hi = hi if hi is not None else values.max()
    normed = (values - lo) / (hi - lo)
    return DOT_SIZE[0] + np.power(normed, power) * (DOT_SIZE[1] - DOT_SIZE[0]), lo, hi


def place_label(ax, x, y, text, placed, x_lim, y_lim, padding=0.02):
    tw = len(text) * 0.02
    th = 0.05
    for angle in np.linspace(0, 2*np.pi, 32, endpoint=False):
        for dist in [0.8, 0.6, 0.4, 0.3]:
            dx, dy = dist * np.cos(angle), dist * np.sin(angle)
            tx, ty = x + dx, y + dy
            margin = 0.1
            if not (x_lim[0]+margin < tx < x_lim[1]-margin and
                    y_lim[0]+margin < ty < y_lim[1]-margin):
                continue
            box = [tx-tw/2-padding, ty-th/2-padding,
                   tx+tw/2+padding, ty+th/2+padding]
            if not any(b[0]<box[2] and box[0]<b[2] and
                       b[1]<box[3] and box[1]<b[3] for b in placed):
                ax.add_line(plt.Line2D([x, tx], [y, ty],
                            color='#333333', linewidth=0.6, zorder=9))
                ax.text(tx, ty, text, fontsize=7, color='#333333',
                        ha='center', va='center', zorder=10)
                placed.append(box)
                return


def draw_inset_legend(ax, data, s_lo, s_hi):
    qs = [data[SIZE_COL].quantile(q) for q in LEGEND_QUANTILES]
    legend_sizes, *_ = scale_sizes(pd.Series(qs), power=0.8, lo=s_lo, hi=s_hi)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    xw = x_lim[1] - x_lim[0]
    yh = y_lim[1] - y_lim[0]

    box_right  = x_lim[1] - 0.02 * xw
    box_top    = y_lim[1] - 0.02 * yh
    box_width  = 0.28 * xw
    box_height = 0.22 * yh

    box_left   = box_right - box_width
    box_bottom = box_top   - box_height

    rect = mpl.patches.FancyBboxPatch(
        (box_left, box_bottom), box_width, box_height,
        boxstyle='round,pad=0.01',
        linewidth=0.6, edgecolor='#bbbbbb',
        facecolor='white', alpha=0.85, zorder=15
    )
    ax.add_patch(rect)

    title_x = box_left + box_width / 2
    title_y = box_top  - 0.018 * yh
    ax.text(title_x, title_y, 'Sample Size',
            fontsize=7.5, fontweight='bold', color='#333333',
            ha='center', va='top', zorder=16)

    n = len(qs)
    row_h = (box_height - 0.045 * yh) / n

    dot_color_rgba  = mpl.colors.to_rgba(DOT_COLOR, alpha=DOT_ALPHA)
    edge_color_rgba = mpl.colors.to_rgba('#333333', alpha=EDGE_ALPHA)

    for i, (sz, label, q_val) in enumerate(zip(legend_sizes, LEGEND_LABELS, qs)):
        cy = box_top - 0.045 * yh - (i + 0.5) * row_h
        cx = box_left + 0.22 * box_width

        ax.scatter([cx], [cy],
                   s=sz,
                   c=[dot_color_rgba],
                   edgecolors=[edge_color_rgba],
                   linewidths=EDGE_WIDTH,
                   zorder=17,
                   clip_on=False)

        ax.text(box_left + 0.38 * box_width, cy,
                f'{label}: {int(q_val)}',
                fontsize=6.5, color='#333333',
                ha='left', va='center', zorder=17)


def make_figure(data, embedding):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sizes, s_lo, s_hi = scale_sizes(data[SIZE_COL])

    top_n_idx = set(
        data.sort_values(SIZE_COL, ascending=False).head(ANNOTATE_TOP_N).index
    )

    for i, (idx, row) in enumerate(data.iterrows()):
        is_top = idx in top_n_idx
        ec = mpl.colors.to_rgba('#333333', alpha=EDGE_ALPHA_TOP if is_top else EDGE_ALPHA)
        ax.scatter(embedding[i, 0], embedding[i, 1],
                   c=mpl.colors.to_rgba(DOT_COLOR, alpha=DOT_ALPHA),
                   s=sizes.iloc[i],
                   edgecolors=[ec],
                   linewidths=EDGE_WIDTH_TOP if is_top else EDGE_WIDTH,
                   zorder=5 if is_top else 4)

    ax.set_xlabel('UMAP 1', fontsize=9, labelpad=8)
    ax.set_ylabel('UMAP 2', fontsize=9, labelpad=8)
    ax.tick_params(labelsize=8, pad=4)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    placed = []

    for _, row in data.sort_values(SIZE_COL, ascending=False).head(ANNOTATE_TOP_N).iterrows():
        i = data.index.get_loc(row.name)
        place_label(ax, embedding[i, 0], embedding[i, 1],
                    row['City'], placed, x_lim, y_lim)

    extreme = pd.concat([
        data.nlargest(ANNOTATE_EXTREME, DEPRESSION_COL),
        data.nsmallest(ANNOTATE_EXTREME, DEPRESSION_COL)
    ])
    for _, row in extreme.iterrows():
        i = data.index.get_loc(row.name)
        place_label(ax, embedding[i, 0], embedding[i, 1],
                    f"{row['City']}: {row[DEPRESSION_COL]:.2f}",
                    placed, x_lim, y_lim)

    draw_inset_legend(ax, data, s_lo, s_hi)

    plt.tight_layout()
    return fig


def main():
    data = pd.read_csv(INPUT_CSV, encoding='utf-8')
    data = data[data[SIZE_COL] > MIN_SAMPLES].copy()
    print(f"City count: {len(data)}")

    setup_style()

    ig_cols = [c for c in data.columns if c.startswith('Avg_IG_Value_')]
    print(f"IG cols: {len(ig_cols)}")
    if len(ig_cols) < 2:
        raise ValueError("IG cols not enough")

    X = StandardScaler().fit_transform(np.nan_to_num(data[ig_cols].values))

    raw_emb = umap.UMAP(**UMAP_PARAMS, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
                        random_state=SEED).fit_transform(X)
    emb = transform_embedding(raw_emb)
    data[['UMAP_IG_1', 'UMAP_IG_2']] = emb

    fig = make_figure(data, emb)
    out_path = f'{OUTPUT_DIR}/{OUTPUT_NAME}'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == '__main__':
    main()