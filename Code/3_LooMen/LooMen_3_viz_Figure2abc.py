import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, MaxNLocator, AutoMinorLocator, ScalarFormatter, FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import torch

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ============================== CONFIG ==============================

# --- input ---
DATAPATH = '../../Data/LooMen'
IG_FILE  = f'{DATAPATH}/Data_2_IG_results.npz'
CSV_FILE = f'{DATAPATH}/Data_3_POI_category.csv'

# --- output ---
CURVE_ROOT = '../../Results/LooMen/IG_explain_full'
FIGURE_DIR = '../../Results/Figure2'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- GP regression: shared params ---
GP_SAMPLE_SIZE = 100000
GP_MAX_POINTS  = 5000
GP_NOISE       = 0.02

# --- GP regression ---
GP_RUNS = {
    'L1': {'percentile': 0.05, 'length_scale': 0.8, 'curve_dir': f'{CURVE_ROOT}/Curve'},
    'L2': {'percentile': 0.03, 'length_scale': 0.5, 'curve_dir': f'{CURVE_ROOT}/Curve_L2'},
}

# --- Figure 2a ---
BOX_TOP_N          = 20
BOX_WIDTH          = 0.65
BOX_FILL_ALPHA     = 0.6
SCATTER_SAMPLE     = 80
SCATTER_ALPHA      = 0.3
SCATTER_SIZE       = 15
JITTER_WIDTH       = 0.2
PLOT_MARGIN        = 0.8
BOX_FIG_W          = 5
BOX_FIG_H          = 2.3
BOX_DPI            = 600
BOX_COLOR_GRADIENT = "#7456a5, #8f77b6, #a998c7, #c5bbd7, #ddebe4, #b6cfc2, #90b4a2, #6a9982, #457f63"

# --- Figure 2b ---
AGG_METHOD           = 'mean'
AGG_SHOW_NAMES       = True
AGG_SORT_MEDIAN      = True
AGG_CURVE_WIDTH      = 1.0
AGG_DISPLAY_SPECIFIC = True
AGG_COLOR            = "#624296, #876BB4, #9a8ec1, #c4bcd4, #b6c7be, #90ae9f, #6CA389, #317052"
AGG_SPECIFIC_CURVES  = ['Residential', 'Work', 'Education', 'Culture', 'Public Space', 'Leisure', 'Dining', 'Medical']
AGG_FIG_W            = 6
AGG_FIG_H            = 4
AGG_DPI              = 600

# --- Figure 2c ---
SHAPE_FEATURES = {
    'U': ["Railway Station", "Game Arcade", "Highway", "Pharmacy"],
    'Inverted_U': ["Library", "Airport", "Sports Venue", "Print Shop", "Bathhouse"],
    'J': ["Village", "Dance Studio", "Shop", "Logistics Company", "Green Space"],
    'Inverted_J': ["Middle School", "Ticket Office", "Amusement Park", "Photo Studio"],
    'S': ["Convenience Store", "Shopping Center", "Museum", "Cafe", "Temple"],
    'Inverted_S': ["Parking Lot", "Scenic Area", "Supermarket", "Bar", "Hospital"],
}

# per-panel axis tweak added on top of auto range: 'shape': [x_delta, y_delta]
AXIS_DELTA = {
    'U':          [0.0,  0.02],
    'Inverted_U': [0.0,  0.02],
    'J':          [0.02, 0.02],
    'Inverted_J': [0.0,  0.02],
    'S':          [0.0,  0.02],
    'Inverted_S': [0.0,  0.02],
}

PANEL_ROWS      = 3
PANEL_COLS      = 2
PANEL_W         = 8
PANEL_H         = 9
PANEL_COLOR     = "#B33B48, #CE636E, #EBEBEB, #4C7DC6, #2057A7"
PANEL_CURVE_W   = 0.9
PANEL_SHOW_NAME = True
X_PADDING_RATIO = 0.05
Y_PADDING_RATIO = 0.05
HSPACE          = 0.2
WSPACE          = 0.2
PANEL_DPI       = 600

# ============================== UTILS ==============================

def parse_colors(s):
    return [c.strip() for c in s.split(',')]

def make_cmap(s, n=256):
    return LinearSegmentedColormap.from_list("cg", parse_colors(s), N=n)

def setup_style_main():
    mpl.rcdefaults()
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.unicode_minus': False,
        'font.size': 5, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.grid': False, 'axes.linewidth': 0.5, 'axes.edgecolor': '#333333',
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3, 'ytick.minor.width': 0.3,
        'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 4,
        'ytick.minor.visible': True, 'xtick.minor.visible': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.transparent': False, 'savefig.pad_inches': 0.1,
        'legend.fontsize': 7, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.axisbelow': True,
    })

def setup_style_panel():
    mpl.rcdefaults()
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.unicode_minus': False,
        'font.size': 5, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.grid': False, 'axes.linewidth': 0.5, 'axes.edgecolor': '#333333',
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.transparent': False, 'savefig.pad_inches': 0.1,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.axisbelow': True,
    })

def curve_npz_path(run_key):
    return os.path.join(GP_RUNS[run_key]['curve_dir'], 'feature_curves_pytorch_gp.npz')

def load_curves(run_key):
    data = np.load(curve_npz_path(run_key), allow_pickle=True)
    feature_names = data['feature_names'].tolist()
    curves = {}
    for i, fn in enumerate(feature_names):
        try:
            curves[fn] = {k: data[f'{k}_{i}'] for k in ['x_curve', 'y_curve', 'y_lower', 'y_upper']}
        except KeyError:
            pass
    return feature_names, curves

def format_axes_curve(ax, x0, x1, y0, y1, xlabel="LooM", ylabel="Attribution"):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='minor', length=4, width=0.6, direction='out')
    ax.tick_params(which='major', length=7, width=0.8, direction='out')
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def format_axes_panel(ax, x0, x1, y0, y1):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='minor', length=3, width=0.5, direction='out')
    ax.tick_params(which='major', length=5, width=0.7, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ============================== GP REGRESSION ==============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RBFKernel:
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        X1, X2 = X1.view(X1.size(0), -1), X2.view(X2.size(0), -1)
        dist = (torch.sum(X1**2, dim=1, keepdim=True)
                + torch.sum(X2**2, dim=1, keepdim=True).T
                - 2 * torch.matmul(X1, X2.T))
        return torch.exp(-0.5 * torch.clamp(dist, min=0) / self.length_scale**2)


class SimpleGP:
    def __init__(self, length_scale=1.0, noise=0.1):
        self.kernel = RBFKernel(length_scale)
        self.noise = noise
        self.alpha = self._L = self.K_inv = self.X_train = None

    def fit(self, X, y):
        self.X_train = X
        K = self.kernel(X) + torch.eye(len(X), device=X.device) * self.noise
        try:
            self._L = torch.linalg.cholesky(K)
            self.alpha = torch.cholesky_solve(y.unsqueeze(1), self._L).squeeze()
            self.K_inv = None
        except Exception:
            self.K_inv = torch.pinverse(K)
            self.alpha = torch.matmul(self.K_inv, y)
            self._L = None

    def predict(self, X_test):
        K_s = self.kernel(X_test, self.X_train)
        mean = torch.matmul(K_s, self.alpha)
        K_ss = self.kernel(X_test) + torch.eye(len(X_test), device=X_test.device) * self.noise
        if self.K_inv is not None:
            var = K_ss - torch.matmul(K_s, torch.matmul(self.K_inv, K_s.t()))
        else:
            v = torch.cholesky_solve(K_s.t(), self._L)
            var = K_ss - torch.matmul(K_s, v)
        std = torch.sqrt(torch.clamp(torch.diag(var), min=0))
        return mean, std


def compute_curves(run_key):
    cfg = GP_RUNS[run_key]
    percentile, length_scale = cfg['percentile'], cfg['length_scale']
    os.makedirs(cfg['curve_dir'], exist_ok=True)

    ig_results = np.load(IG_FILE, allow_pickle=True)
    feature_ig_array = ig_results['feature_ig_array']
    feature_names = ig_results['feature_names'].tolist()
    sampled_features = ig_results['feature_values']

    if sampled_features.shape[0] > GP_SAMPLE_SIZE:
        np.random.seed(RANDOM_SEED)
        idx = np.random.choice(sampled_features.shape[0], GP_SAMPLE_SIZE, replace=False)
        sampled_features, feature_ig_array = sampled_features[idx], feature_ig_array[idx]

    curve_data, valid_names, perf_list = {}, [], []

    for i, feature in enumerate(tqdm(feature_names, desc=f"GP fitting [{run_key}]", ncols=80)):
        fidx = feature_names.index(feature)
        fx, fy = sampled_features[:, fidx], feature_ig_array[:, fidx]

        lx, ux = np.percentile(fx, percentile * 100), np.percentile(fx, (1 - percentile) * 100)
        ly, uy = np.percentile(fy, percentile * 100), np.percentile(fy, (1 - percentile) * 100)
        mask = (fx >= lx) & (fx <= ux) & (fy >= ly) & (fy <= uy)
        fx, fy = fx[mask], fy[mask]
        if len(fx) <= 10:
            continue

        sidx = np.argsort(fx)
        xs, ys = fx[sidx], fy[sidx]
        x_smooth = np.linspace(xs.min(), xs.max(), 200)

        if len(xs) > GP_MAX_POINTS:
            ii = np.linspace(0, len(xs) - 1, GP_MAX_POINTS, dtype=int)
            xt, yt = xs[ii], ys[ii]
        else:
            xt, yt = xs, ys

        xt_t = torch.tensor(xt, dtype=torch.float32).view(-1, 1).to(device)
        yt_t = torch.tensor(yt, dtype=torch.float32).to(device)
        xs_t = torch.tensor(x_smooth, dtype=torch.float32).view(-1, 1).to(device)

        gp = SimpleGP(length_scale=length_scale, noise=GP_NOISE)
        gp.fit(xt_t, yt_t)
        mean, std = gp.predict(xs_t)
        yp = mean.cpu().numpy()

        tp, _ = gp.predict(xt_t)
        perf_list.append({
            'MSE':  mean_squared_error(yt, tp.cpu().numpy()),
            'RMSE': np.sqrt(mean_squared_error(yt, tp.cpu().numpy())),
            'MAE':  mean_absolute_error(yt, tp.cpu().numpy()),
            'R2':   r2_score(yt, tp.cpu().numpy()),
        })

        curve_data[feature] = {
            'x_curve': x_smooth, 'y_curve': yp,
            'y_lower': (mean - 1.96 * std).cpu().numpy(),
            'y_upper': (mean + 1.96 * std).cpu().numpy(),
        }
        valid_names.append(feature)

        if (i + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg = {k: np.mean([p[k] for p in perf_list]) for k in ['MSE', 'RMSE', 'MAE', 'R2']}

    save_dict = {
        'feature_names': np.array(valid_names, dtype=object),
        'estimation_method': 'pytorch_gp',
        'avg_performance_metrics': avg,
        'global_x_range': np.array([min(curve_data[f]['x_curve'].min() for f in valid_names),
                                     max(curve_data[f]['x_curve'].max() for f in valid_names)]),
        'global_y_range': np.array([min(curve_data[f]['y_curve'].min() for f in valid_names),
                                     max(curve_data[f]['y_curve'].max() for f in valid_names)]),
    }
    for i, f in enumerate(valid_names):
        for k in ['x_curve', 'y_curve', 'y_lower', 'y_upper']:
            save_dict[f'{k}_{i}'] = curve_data[f][k]

    np.savez(curve_npz_path(run_key), **save_dict)
    print(f"[{run_key}] Curve={len(valid_names)} | R2={avg['R2']:.4f} | RMSE={avg['RMSE']:.6f} | MAE={avg['MAE']:.6f}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================== FIGURE 2A ==============================

def figure2a_boxplot():
    setup_style_main()
    os.makedirs(FIGURE_DIR, exist_ok=True)

    loaded = np.load(IG_FILE, allow_pickle=True)
    feature_ig_array = loaded['feature_ig_array']
    feature_names = loaded['feature_names'].tolist()
    ig_df = pd.DataFrame(feature_ig_array, columns=feature_names)

    top_features = ig_df.median().abs().sort_values(ascending=False).index[:BOX_TOP_N]
    selected = ig_df[top_features].median().abs().sort_values(ascending=False).index.tolist()

    cmap = make_cmap(BOX_COLOR_GRADIENT, n=len(selected))
    n = len(selected)

    wmins, wmaxs = [], []
    for f in selected:
        v = ig_df[f].dropna().values
        q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
        iqr = q3 - q1
        wmins.append(np.min(v[v >= q1 - 1.5 * iqr]))
        wmaxs.append(np.max(v[v <= q3 + 1.5 * iqr]))
    dmin, dmax = min(wmins), max(wmaxs)
    dr = dmax - dmin
    pad = dr * 0.04
    raw_step = dr / 6
    mag = 10 ** np.floor(np.log10(raw_step))
    for c in [1, 2, 2.5, 5, 10]:
        step = c * mag
        if dr / step <= 8:
            break
    major_step = step
    y_min = np.floor((dmin - pad) / major_step) * major_step
    y_max = np.ceil((dmax + pad) / major_step) * major_step

    np.random.seed(RANDOM_SEED)
    sidx = (np.random.choice(len(ig_df), SCATTER_SAMPLE, replace=False)
            if len(ig_df) > SCATTER_SAMPLE else np.arange(len(ig_df)))

    fig, ax = plt.subplots(figsize=(BOX_FIG_W, BOX_FIG_H + 0.5), dpi=BOX_DPI)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.2, alpha=0.9, zorder=0)

    for i, f in enumerate(selected):
        vals = ig_df.iloc[sidx][f].values
        jitter = np.random.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(vals))
        color = cmap(i / (n - 1) if n > 1 else 0)
        ax.scatter([i + j for j in jitter], vals, alpha=SCATTER_ALPHA,
                   s=SCATTER_SIZE, color=color, edgecolors='none', marker='.', zorder=1)

    bp = ax.boxplot([ig_df[f].values for f in selected],
                    positions=list(range(n)), widths=BOX_WIDTH,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=0.5, edgecolor="#333333"),
                    whiskerprops=dict(linewidth=0.5, color="#333333"),
                    capprops=dict(linewidth=0.5, color="#333333"),
                    medianprops=dict(color="#333333", linewidth=0.5, solid_capstyle='butt'),
                    zorder=2)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor((*to_rgb(cmap(i / (n - 1) if n > 1 else 0)), BOX_FILL_ALPHA))

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel('Integrated Gradients Value', fontsize=7, labelpad=4)
    ax.set_xlim(-PLOT_MARGIN, n - 1 + PLOT_MARGIN / 2)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(selected, rotation=90, fontsize=7, ha='center')
    sci = ScalarFormatter(useMathText=True)
    sci.set_scientific(True)
    sci.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(sci)
    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(major_step / 2))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(axis='both', which='major', direction='out', length=3.0, width=0.5,
                   color='#333333', labelsize=7, bottom=True, left=True)
    ax.tick_params(axis='y', which='minor', direction='out', length=1.5, width=0.3,
                   color='#333333', bottom=False, left=True)
    ax.tick_params(axis='x', which='major', pad=3)
    ax.yaxis.get_offset_text().set_fontsize(7)

    out = os.path.join(FIGURE_DIR, 'Figure2a.png')
    fig.savefig(out, dpi=BOX_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Figure 2a] -> {out}")

# ============================== FIGURE 2B ==============================

def figure2b_primary_category():
    setup_style_main()
    os.makedirs(FIGURE_DIR, exist_ok=True)

    df = pd.read_csv(CSV_FILE, encoding='utf-8')
    category_mapping = {}
    primary_order = []
    for _, row in df.iterrows():
        primary = row['Primary Category'].strip()
        for sec in row['Secondary Category'].strip().split('、'):
            if sec.strip():
                category_mapping[sec.strip()] = primary
        if primary not in primary_order:
            primary_order.append(primary)

    _, curves = load_curves('L1')
    agg_fn = {'mean': np.mean, 'median': np.median, 'max': np.max, 'sum': np.sum}[AGG_METHOD]

    primary_curves = {}
    for cat in primary_order:
        secs = [f for f, p in category_mapping.items() if p == cat and f in curves]
        if not secs:
            continue
        ys = np.array([curves[f]['y_curve'] for f in secs])
        primary_curves[cat] = {
            'x_curve': curves[secs[0]]['x_curve'],
            'y_curve': agg_fn(ys, axis=0),
        }

    sorted_feats = list(primary_curves.keys())
    if AGG_SORT_MEDIAN:
        sorted_feats = sorted(sorted_feats,
                              key=lambda f: np.median(primary_curves[f]['y_curve']), reverse=True)
    if AGG_DISPLAY_SPECIFIC:
        sorted_feats = [f for f in AGG_SPECIFIC_CURVES if f in primary_curves]

    all_x = [v for c in primary_curves.values() for v in c['x_curve']]
    all_y = [v for c in primary_curves.values() for v in c['y_curve']]
    xm, ym = 0.05 * (max(all_x) - min(all_x)), 0.05 * (max(all_y) - min(all_y))
    x0, x1 = min(all_x) - xm, max(all_x) + xm
    y0, y1 = min(all_y) - ym, max(all_y) + ym

    fig, ax = plt.subplots(figsize=(AGG_FIG_W, AGG_FIG_H))
    ax.axhline(0, color='k', linestyle='--', alpha=1.0, linewidth=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.3)

    color_map = dict(zip(AGG_SPECIFIC_CURVES, parse_colors(AGG_COLOR)))
    for feat in sorted_feats:
        c = primary_curves[feat]
        color = color_map.get(feat, '#333333')
        ax.plot(c['x_curve'], c['y_curve'], color=color, linewidth=AGG_CURVE_WIDTH)
        if AGG_SHOW_NAMES:
            ax.text(c['x_curve'][-1] + 0.02 * (x1 - x0), c['y_curve'][-1],
                    feat, color='k', fontsize=9, va='center', ha='left')

    format_axes_curve(ax, x0, x1, y0, y1)
    out = os.path.join(FIGURE_DIR, 'Figure2b.png')
    fig.savefig(out, dpi=AGG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Figure 2b] -> {out}")

# ============================== FIGURE 2C ==============================

def figure2c_shape_panel():
    setup_style_panel()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    _, curves = load_curves('L2')

    cmap = make_cmap(PANEL_COLOR)
    fig, axes = plt.subplots(PANEL_ROWS, PANEL_COLS, figsize=(PANEL_W, PANEL_H), dpi=PANEL_DPI)

    for idx, shape_name in enumerate(SHAPE_FEATURES.keys()):
        ax = axes[idx // PANEL_COLS, idx % PANEL_COLS]
        valid_feats = [f for f in SHAPE_FEATURES[shape_name] if f in curves]
        for f in SHAPE_FEATURES[shape_name]:
            if f not in curves:
                print(f"  [Warning] feature not found in data: {f}")

        ax.axhline(0, color='k', linestyle='--', alpha=0.7, linewidth=0.4)
        ax.axvline(0, color='k', linestyle='--', alpha=0.7, linewidth=0.4)

        if not valid_feats:
            format_axes_panel(ax, -1, 1, -1, 1)
            continue

        xs = np.concatenate([curves[f]['x_curve'] for f in valid_feats])
        ys = np.concatenate([curves[f]['y_curve'] for f in valid_feats])

        xm = X_PADDING_RATIO * (xs.max() - xs.min()) if xs.max() > xs.min() else 0.1
        ym = Y_PADDING_RATIO * (ys.max() - ys.min()) if ys.max() > ys.min() else 0.1
        x_delta, y_delta = AXIS_DELTA.get(shape_name, [0.0, 0.0])

        x0, x1 = xs.min() - xm - x_delta, xs.max() + xm + x_delta
        y0, y1 = ys.min() - ym - y_delta, ys.max() + ym + y_delta

        norm = plt.Normalize(-max(abs(x0), abs(x1)), max(abs(x0), abs(x1)))
        for feat in valid_feats:
            c = curves[feat]
            pts = np.array([c['x_curve'], c['y_curve']]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=PANEL_CURVE_W)
            lc.set_array((pts[:-1, 0, 0] + pts[1:, 0, 0]) / 2)
            ax.add_collection(lc)
            if PANEL_SHOW_NAME:
                ax.text(c['x_curve'][-1] + 0.01 * (x1 - x0), c['y_curve'][-1],
                        feat, color='k', fontsize=8, va='center', ha='left')

        format_axes_panel(ax, x0, x1, y0, y1)

    for row in range(PANEL_ROWS):
        for col in range(PANEL_COLS):
            if row == PANEL_ROWS - 1:
                axes[row, col].set_xlabel("LooM", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel("Attribution", fontsize=9)

    plt.tight_layout()
    fig.subplots_adjust(hspace=HSPACE, wspace=WSPACE)
    out = os.path.join(FIGURE_DIR, 'Figure2c.png')
    fig.savefig(out, dpi=PANEL_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[Figure 2c] -> {out}")


if __name__ == "__main__":
    compute_curves('L1')
    compute_curves('L2')
    figure2a_boxplot()
    figure2b_primary_category()
    figure2c_shape_panel()