import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import linregress
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter, LinearLocator
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature

warnings.filterwarnings("ignore")

# =============================
INPUT_CSV        = '../../Data/City/City2.csv'
POP_CSV          = '../../Data/City/Population.csv'
VALID_CITIES_CSV = '../../Data/City/City_name.csv'
CITY_GEOJSON     = '../../Data/City/City.geojson'
PROV_GEOJSON     = '../../Data/City/Province.geojson'

OUTPUT_DIR   = '../../Results/Figure4'
OUTPUT_NAME  = '{feature}.png'

FEATURES = ["Pharmacy", "Supermarket", "Bank", "Green Space", "Middle School", "Cafe"]

FONT_SIZE_COLORBAR = 12
FONT_SIZE_FEATURE  = 14
FONT_SIZE_SCALEBAR = 12
FONT_SIZE_NORTH    = 14

FIG_SIZE  = (10, 8)
INSET_POS = [0.75, 0.10, 0.15, 0.20]
CBAR_POS  = [0.14, 0.13, 0.34, 0.02]

SCALEBAR_X_REL      = 0.15
SCALEBAR_Y_REL      = 0.15
FEATURE_LABEL_X_REL = 0.12
FEATURE_LABEL_Y_REL = 0.95
NORTH_X_REL         = 0.85
NORTH_Y_REL         = 0.90

BLUE  = "#2E69AF"
WHITE = "#FFFFFF"
RED   = "#BF5262"
# ============================

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

STATIC_CMAP = LinearSegmentedColormap.from_list("diverging", [BLUE, WHITE, RED])
BASE_COLS = ['City', 'City_name', 'Population', 'Sample_Count']


def normalize_name(name):
    if isinstance(name, str):
        return name.replace('市', '').replace('地区', '').replace('自治州', '').replace('盟', '').strip()
    return name


def calculate_hbm(df, feature):
    col_val   = f"Avg_IG_Value_{feature}"
    col_lower = f"Avg_IG_CI_Lower_{feature}"
    col_upper = f"Avg_IG_CI_Upper_{feature}"

    if col_val not in df.columns:
        return np.zeros(len(df)), np.zeros(len(df))

    y_i      = df[col_val].values
    ci_width = np.abs(df[col_upper].values - df[col_lower].values)
    se_i     = np.where(ci_width == 0, 1e-6, ci_width / 3.92)
    sigma_stat_sq = se_i ** 2

    log_pop    = np.log10(df['Population'].values)
    log_sample = np.log10(df['Sample_Count'].values)
    slope, intercept, _, _, _ = linregress(log_pop, log_sample)

    deficit       = np.maximum(0, -(log_sample - (slope * log_pop + intercept)))
    sigma_rep_sq  = (deficit * np.mean(se_i)) ** 2

    weights_pre = 1.0 / (sigma_stat_sq + 1e-9)
    mu_global   = np.average(y_i, weights=weights_pre)
    tau_sq      = max(1e-6, np.var(y_i, ddof=1) - np.mean(sigma_stat_sq))

    sigma_total_sq = sigma_stat_sq + sigma_rep_sq
    W = tau_sq / (tau_sq + sigma_total_sq)
    return W * y_i + (1 - W) * mu_global, W


def get_symmetric_limit(data, col):
    vmin, vmax = np.percentile(data[col], 5), np.percentile(data[col], 95)
    limit = max(abs(vmin), abs(vmax))
    return limit if limit > 1e-6 else 0.001


def format_sci_tick(x, pos=None):
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    abs_x = abs(x)
    exp = int(np.floor(np.log10(abs_x)))
    mant = abs_x / 10**exp
    sup_map = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp_str = str(exp).translate(sup_map)
    return f"{sign}{mant:.0f}×10{exp_str}"


def add_north_arrow(ax, x, y, size):
    ax.text(x, y + size / 2 + 30000, 'N', ha='center', va='bottom',
            fontsize=FONT_SIZE_NORTH, fontweight='bold', zorder=10)
    path_data = [(0.0, -0.5), (0.4, -0.4), (0.0, 0.5),
                 (0.0, -0.5), (-0.4, -0.4), (0.0, 0.5)]
    codes = [mpatches.Path.MOVETO, mpatches.Path.LINETO, mpatches.Path.LINETO] * 2
    verts_right = [(x + dx * size * 0.6, y + dy * size) for dx, dy in path_data[:3]]
    verts_left  = [(x + dx * size * 0.6, y + dy * size) for dx, dy in path_data[3:]]
    ax.add_patch(mpatches.PathPatch(mpatches.Path(verts_right, codes[:3]),
                                    facecolor='black', edgecolor='black', lw=0.5, zorder=10))
    ax.add_patch(mpatches.PathPatch(mpatches.Path(verts_left,  codes[3:]),
                                    facecolor='white', edgecolor='black', lw=0.5, zorder=10))


def add_scalebar(ax, x, y):
    ticks_km = [0, 300, 600, 1200]
    ticks_m  = [t * 1000 for t in ticks_km]
    ax.plot([x, x + ticks_m[-1]], [y, y], color='black', linewidth=1.2, zorder=10)
    for i, dist_m in enumerate(ticks_m):
        tick_x = x + dist_m
        ax.plot([tick_x, tick_x], [y, y + 40000], color='black', linewidth=1.2, zorder=10)
        label = f"{ticks_km[i]}km" if i == len(ticks_km) - 1 else str(ticks_km[i])
        ax.text(tick_x, y + 60000, label, ha='center', va='bottom',
                fontsize=FONT_SIZE_SCALEBAR, zorder=10)


def plot_static(data, col, prov_map, out_dir):
    limit = get_symmetric_limit(data, col)
    norm  = Normalize(vmin=-limit, vmax=limit)
    proj  = ccrs.LambertConformal(central_longitude=105, central_latitude=35,
                                  standard_parallels=(25, 47))

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_extent([73, 138, 16, 54], crs=ccrs.PlateCarree())

    ax.add_feature(ShapelyFeature(prov_map.geometry, ccrs.PlateCarree(),
                                  edgecolor='#333333', facecolor='none', linewidth=0.4, zorder=3))
    for _, row in data.iterrows():
        ax.add_feature(ShapelyFeature([row.geometry], ccrs.PlateCarree(),
                                      facecolor=STATIC_CMAP(norm(row[col])),
                                      edgecolor='none', zorder=1))
    ax.add_feature(ShapelyFeature(data.geometry, ccrs.PlateCarree(),
                                  edgecolor='#555555', facecolor='none',
                                  linewidth=0.05, alpha=0.3, zorder=2))
    ax.axis('off')

    x0, x1, y0, y1 = ax.get_extent()
    ax.text(x0 + FEATURE_LABEL_X_REL * (x1 - x0), y0 + FEATURE_LABEL_Y_REL * (y1 - y0),
            col, ha='left', va='top', fontsize=FONT_SIZE_FEATURE, fontweight='bold', zorder=10)
    add_north_arrow(ax, x0 + NORTH_X_REL * (x1 - x0), y0 + NORTH_Y_REL * (y1 - y0),
                    0.05 * (y1 - y0))
    add_scalebar(ax, x0 + SCALEBAR_X_REL * (x1 - x0), y0 + SCALEBAR_Y_REL * (y1 - y0))

    ax_sub = fig.add_axes(INSET_POS, projection=proj)
    ax_sub.set_extent([105, 120, 2, 26], crs=ccrs.PlateCarree())
    ax_sub.add_feature(ShapelyFeature(prov_map.geometry, ccrs.PlateCarree(),
                                      edgecolor='#333333', facecolor='none', linewidth=0.4, zorder=3))
    for _, row in data[data.geometry.bounds.miny <= 26].iterrows():
        ax_sub.add_feature(ShapelyFeature([row.geometry], ccrs.PlateCarree(),
                                          facecolor=STATIC_CMAP(norm(row[col])),
                                          edgecolor='none', zorder=1))

    cax = fig.add_axes(CBAR_POS)
    cb  = fig.colorbar(ScalarMappable(norm=norm, cmap=STATIC_CMAP), cax=cax,
                       orientation='horizontal', extend='both')
    cb.ax.xaxis.set_major_locator(LinearLocator(5))
    cb.ax.xaxis.set_major_formatter(FuncFormatter(format_sci_tick))
    cb.ax.tick_params(labelsize=FONT_SIZE_COLORBAR, length=3, pad=2)
    cax.set_title('Attribution', fontsize=FONT_SIZE_COLORBAR, fontfamily='Arial', pad=6)

    out_path = f"{out_dir}/{OUTPUT_NAME.format(feature=col)}"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"  saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pop_data = pd.read_csv(POP_CSV)
    pop_data['City_name'] = pop_data['name'].apply(normalize_name)
    pop_data['Population'] = pd.to_numeric(pop_data['人口总数_平均'], errors='coerce')
    valid_cities_set = set(pd.read_csv(VALID_CITIES_CSV)['City'])

    city_map = gpd.read_file(CITY_GEOJSON)
    prov_map = gpd.read_file(PROV_GEOJSON)
    city_map['City_name'] = city_map['name'].apply(normalize_name)

    csv_data = pd.read_csv(INPUT_CSV)
    csv_data['City_name'] = csv_data['City'].apply(normalize_name)
    merged = csv_data.merge(pop_data[['City_name', 'Population']], on='City_name')
    final_data = merged[merged['City'].isin(valid_cities_set)].copy()
    print(f"Valid cities: {len(final_data)}")

    for feature in FEATURES:
        posterior, _ = calculate_hbm(final_data, feature)
        final_data[feature] = posterior

    df = final_data[BASE_COLS + FEATURES]
    gdf = city_map.merge(df, on='City_name', how='inner')

    for feature in FEATURES:
        plot_static(gdf, feature, prov_map, OUTPUT_DIR)

    print("Done")


if __name__ == "__main__":
    main()