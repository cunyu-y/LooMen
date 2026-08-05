import os

import numpy as np
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

# ============================================================

NPZ_FILE     = "../../Data/LooMen/Data_2_IG_results.npz"
PARQUET_FILE = "../../Data/Cluster/Cluster_result.parquet"

SAVE_DIR      = "../../Results/Figure3"
IG_FIG_NAME   = "Figure3b_IG.png"
LOOM_FIG_NAME = "Figure3b_LooM.png"

FIG_DPI      = 300
ROTATE_90_CW = True

OFFSET = 0.02

IG_TOP_CLUSTERS    = 8
IG_N_TOP_FEATURES  = 88
IG_MAX_FEATURES    = 100
IG_ENHANCE         = 0.7
IG_INNER_RADIUS    = 0.7
IG_OUTER_RADIUS    = 0.9
IG_COLOR_GRADIENT  = "#26587C, #38688C, #4F7EA2, #6391B4, #FFFFFF, #B6A3D4, #9B83C2, #7456A5, #644499"

LOOM_TOP_CLUSTERS  = 24
LOOM_ENHANCE       = 1.0
LOOM_INNER_RADIUS  = 0.3
LOOM_OUTER_RADIUS  = 0.9
LOOM_COLOR_GRADIENT = "#A12633, #C0525E, #FFFFFF, #5C977A, #266848"

MIN_ABS_THRESH = 0.000
# ============================================================

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
plt.rcParams['font.sans-serif']    = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


class PowerNorm(Normalize):
    def __init__(self, vmin=None, vmax=None, enhance_factor=0.3, clip=False):
        super().__init__(vmin, vmax, clip)
        self.enhance_factor = enhance_factor

    def __call__(self, value, clip=None):
        v = np.array(value, copy=True)
        v = np.sign(v) * (np.abs(v) / self.vmax) ** self.enhance_factor
        return (v + 1) / 2


def load_data():
    data = np.load(NPZ_FILE, allow_pickle=True)
    feature_names = data['feature_names'].tolist()
    ig_values     = np.array(data['feature_ig_array'], dtype=np.float32)
    feat_values   = np.array(data['feature_values'],   dtype=np.float32)

    cluster_df   = pl.read_parquet(PARQUET_FILE).select('cluster')
    clusters_col = cluster_df['cluster'].to_numpy()
    assert ig_values.shape[0] == len(clusters_col)
    assert feat_values.shape[0] == len(clusters_col)

    base = {name: ig_values[:, i] for i, name in enumerate(feature_names)}
    base.update({f"__loom_{name}": feat_values[:, i] for i, name in enumerate(feature_names)})
    base['cluster'] = clusters_col

    df = pl.DataFrame(base).filter(pl.col('cluster') != -1)
    return df, feature_names


def select_clusters(df, n_top):
    counts = (
        df.group_by('cluster')
        .agg(pl.len().alias('count'))
        .with_columns((pl.col('count') / df.height * 100).round(2).alias('pct'))
        .sort('pct', descending=True)
    )
    top = counts.head(n_top)['cluster'].to_list()[::-1]
    return top, counts


def select_ig_features(df, feature_names, clusters):
    means = {}
    for c in clusters:
        sub     = df.filter(pl.col('cluster') == c).select(feature_names)
        means[c] = dict(zip(feature_names, sub.mean().row(0)))

    cluster_cols = [str(c) for c in clusters]
    feature_df   = pl.DataFrame(
        {'feature': feature_names, **{str(c): [means[c][f] for f in feature_names] for c in clusters}}
    )

    important = set()
    for c in clusters:
        col  = str(c)
        vals = feature_df.select(['feature', col]).with_columns(pl.col(col).abs().alias('abs_val'))
        sig  = vals.filter(pl.col('abs_val') > MIN_ABS_THRESH) if MIN_ABS_THRESH > 0 else vals
        if sig.height == 0:
            sig = vals
        top = sig.sort('abs_val', descending=True).head(IG_N_TOP_FEATURES)['feature'].to_list()
        important.update(top)

    plot_data = (
        feature_df.filter(pl.col('feature').is_in(list(important)))
        .with_columns(pl.max_horizontal([pl.col(c).abs() for c in cluster_cols]).alias('_max'))
        .sort('_max', descending=True)
        .drop('_max')
    )

    if len(plot_data) > IG_MAX_FEATURES:
        plot_data = plot_data.head(IG_MAX_FEATURES)

    return plot_data, cluster_cols


def build_loom_plot_data(df, feature_names, clusters, ig_feature_order):
    loom_cols = [f"__loom_{name}" for name in feature_names]
    means = {}
    for c in clusters:
        sub      = df.filter(pl.col('cluster') == c).select(loom_cols)
        row_vals = sub.mean().row(0)
        means[c] = dict(zip(feature_names, row_vals))

    cluster_cols = [str(c) for c in clusters]
    feature_df   = pl.DataFrame(
        {'feature': feature_names, **{str(c): [means[c][f] for f in feature_names] for c in clusters}}
    )

    ordered = feature_df.filter(pl.col('feature').is_in(ig_feature_order))
    order_map = {f: i for i, f in enumerate(ig_feature_order)}
    ordered   = ordered.with_columns(
        pl.col('feature').replace_strict(order_map).alias('_order')
    ).sort('_order').drop('_order')

    return ordered, cluster_cols


def build_colormap(gradient_str):
    colors = [c.strip() for c in gradient_str.split(',')]
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=10000)


def draw_rings(ax, plot_data, cluster_cols, cm, max_abs_val, enhance, inner_r, outer_r):
    n_features     = len(plot_data)
    start_angle    = np.radians(0)
    end_angle      = np.radians(270)
    feature_angles = np.linspace(start_angle, end_angle, n_features, endpoint=False)
    feature_width  = (end_angle - start_angle) / n_features
    ring_width     = (outer_r - inner_r) / len(cluster_cols)

    for i, col in enumerate(cluster_cols):
        r_inner = inner_r + i * ring_width
        r_outer = r_inner + ring_width
        values  = plot_data[col].to_numpy()
        for j in range(n_features):
            val       = values[j]
            norm_val  = np.sign(val) * (np.abs(val) / max_abs_val) ** enhance
            color_val = (norm_val + 1) / 2
            theta      = np.linspace(feature_angles[j], feature_angles[j] + feature_width, 50)
            theta_path = np.concatenate([theta, theta[::-1]])
            r_path     = np.concatenate([np.full_like(theta, r_outer), np.full_like(theta, r_inner)])
            ax.fill(theta_path, r_path, color=cm(color_val), edgecolor='white', linewidth=0.2)

    theta_full = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta_full, np.full_like(theta_full, outer_r), color='black', linewidth=1.0)

    return feature_angles, feature_width, ring_width


def draw_feature_labels(ax, plot_data, feature_angles, feature_width, outer_r):
    for i, feature in enumerate(plot_data['feature'].to_list()):
        angle     = feature_angles[i] + feature_width / 2
        angle_deg = np.degrees(angle) % 360

        label = feature.split('_')[-1] if '_' in feature else feature
        label = label[:50] + '...' if len(label) > 50 else label

        label_radius = outer_r + OFFSET

        if angle_deg <= 90 or angle_deg > 270:
            ha = 'left'
            rotation = 90 - angle_deg
        else:
            ha = 'right'
            rotation = 270 - angle_deg

        ax.text(angle, label_radius, label,
                ha=ha,
                va='center',
                rotation=rotation,
                rotation_mode='anchor',
                fontsize=10)


def draw_cluster_labels(ax, clusters, cluster_stats, ring_width, inner_r):
    label_angle = np.radians(0)
    for i, cluster in enumerate(clusters):
        pct   = cluster_stats.filter(pl.col('cluster') == cluster)['pct'][0]
        r_mid = inner_r + (i + 0.5) * ring_width
        ax.text(label_angle, r_mid, f"Cluster{cluster} ({pct}%)", ha='right', va='center', fontsize=8)


def draw_colorbar(fig, ax, cm, max_abs_val, enhance, label):
    sm = plt.cm.ScalarMappable(cmap=cm, norm=PowerNorm(-max_abs_val, max_abs_val, enhance_factor=enhance))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
    cbar.set_label(label, fontsize=12)


def setup_polar_ax(ax, outer_r):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.spines['polar'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.set_ylim(0, outer_r + 0.1)


def rotate_clockwise_90(path):
    transpose_enum = getattr(Image, 'Transpose', Image)
    with Image.open(path) as img:
        dpi     = img.info.get('dpi', (FIG_DPI, FIG_DPI))
        rotated = img.transpose(transpose_enum.ROTATE_270)
        rotated.save(path, dpi=dpi)


def plot_both():
    df, feature_names = load_data()

    ig_clusters, ig_stats     = select_clusters(df, IG_TOP_CLUSTERS)
    ig_plot_data, ig_cols     = select_ig_features(df, feature_names, ig_clusters)
    ig_feature_order          = ig_plot_data['feature'].to_list()

    loom_clusters, loom_stats = select_clusters(df, LOOM_TOP_CLUSTERS)
    loom_plot_data, loom_cols = build_loom_plot_data(df, feature_names, loom_clusters, ig_feature_order)

    ig_cm   = build_colormap(IG_COLOR_GRADIENT)
    loom_cm = build_colormap(LOOM_COLOR_GRADIENT)

    ig_max   = max(abs(v) for col in ig_cols   for v in ig_plot_data[col].to_list())
    loom_max = max(abs(v) for col in loom_cols for v in loom_plot_data[col].to_list())

    os.makedirs(SAVE_DIR, exist_ok=True)

    for plot_data, clusters, stats, cols, cm, max_val, enhance, inner_r, outer_r, cbar_label, fig_name in [
        (ig_plot_data,   ig_clusters,   ig_stats,   ig_cols,   ig_cm,   ig_max,
         IG_ENHANCE,   IG_INNER_RADIUS,   IG_OUTER_RADIUS,
         'Integrated Gradients Attribution', IG_FIG_NAME),
        (loom_plot_data, loom_clusters, loom_stats, loom_cols, loom_cm, loom_max,
         LOOM_ENHANCE, LOOM_INNER_RADIUS, LOOM_OUTER_RADIUS,
         'LooM', LOOM_FIG_NAME),
    ]:
        fig = plt.figure(figsize=(14, 14))
        ax  = fig.add_subplot(111, polar=True)

        feature_angles, feature_width, ring_width = draw_rings(
            ax, plot_data, cols, cm, max_val, enhance, inner_r, outer_r
        )
        draw_feature_labels(ax, plot_data, feature_angles, feature_width, outer_r)
        draw_cluster_labels(ax, clusters, stats, ring_width, inner_r)
        draw_colorbar(fig, ax, cm, max_val, enhance, cbar_label)
        setup_polar_ax(ax, outer_r)

        out_path = f"{SAVE_DIR}/{fig_name}"
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close(fig)

        if ROTATE_90_CW:
            rotate_clockwise_90(out_path)

        print(f"Done:{fig_name}")


if __name__ == "__main__":
    plot_both()