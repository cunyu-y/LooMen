import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import os

# ======================== 配置 ========================
CURVE_DATA_DIR = "../../Data/LooMLP/LooM_Curve_full/curve_data"
OUTPUT_DIR     = "../../Results/Figure5"

COLORS_SIM  = ["#2f6a4d","#51846a","#739e88","#95b9a7","#b9d4c7","#E2EAE6"]
COLORS_SENT = ["#5e408f","#947fb6","#ccc2dc","#EBEBEB","#b9d4c7","#739e88","#2f6a4d"]
# ======================================================

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.sans-serif'] = ['Arial','DejaVu Sans','Helvetica']
mpl.rcParams['axes.unicode_minus'] = False
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    sent_curves = np.load(f"{CURVE_DATA_DIR}/all_sent_curves.npz", allow_pickle=True)
    sim_curves  = np.load(f"{CURVE_DATA_DIR}/all_sim_curves.npz",  allow_pickle=True)
    sent_dict = {k: sent_curves[k].item() for k in sent_curves.files}
    sim_dict  = {k: sim_curves[k].item()  for k in sim_curves.files}
    return sent_dict, sim_dict


def plot(sent_dict, sim_dict):
    features = list(sent_dict.keys())
    first    = features[0]

    sim_vals  = np.array(sent_dict[first]['sim_values'])
    sent_vals = np.array(sim_dict[first]['sent_values'])

    sent_percentages = sent_dict[first]['sent_percentages']
    sim_percentages  = sim_dict[first]['sim_percentages']

    cmap_sim  = LinearSegmentedColormap.from_list("sim",  COLORS_SIM,  N=256)
    cmap_sent = LinearSegmentedColormap.from_list("sent", COLORS_SENT, N=256)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), dpi=300)

    for idx, pct in enumerate(reversed(sim_percentages)):
        c   = COLORS_SIM[idx]
        key = f"sim_{pct}"
        curves_list = []
        for feat in features:
            d = sim_dict.get(feat)
            if d is None or key not in d['curves']:
                continue
            y        = np.array(d['curves'][key]['curve'])
            sv       = np.array(d['sent_values'])
            y_interp = np.interp(sent_vals, sv, y)
            curves_list.append(y_interp)
            ax1.plot(sent_vals, y_interp, color=c, alpha=0.04, linewidth=0.5)

        if not curves_list:
            continue
        arr  = np.array(curves_list)
        mean = np.nanmean(arr, axis=0)
        ci   = 1.96 * np.nanstd(arr, axis=0) / np.sqrt(len(arr))
        ax1.fill_between(sent_vals, mean - ci, mean + ci, color=c, alpha=0.15)
        ax1.plot(sent_vals, mean, color=c, linewidth=2.0)

    cax1 = ax1.inset_axes([0.0, 1.01, 1.0, 0.04])
    cb1  = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap_sim.reversed()),
        cax=cax1, orientation='horizontal', ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    cb1.ax.set_xticklabels(['0%','20%','40%','60%','80%','100%'], fontsize=8)
    cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.set_title("Similarity", fontsize=9, pad=10)

    ax1.set_xlabel("Sentiment")
    ax1.set_ylabel("LooM")
    ax1.set_xlim([sent_vals[0], sent_vals[-1]])
    ax1.set_ylim([-1.05, 1.05])
    ax1.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax1.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax1.grid(False)

    for idx, pct in enumerate(reversed(sent_percentages)):
        c = COLORS_SENT[idx]
        curves_list = []
        for feat in features:
            d = sent_dict.get(feat)
            if d is None or pct not in d['curves']:
                continue
            y        = np.array(d['curves'][pct]['curve'])
            sv       = np.array(d['sim_values'])
            y_interp = np.interp(sim_vals, sv, y)
            curves_list.append(y_interp)
            ax2.plot(sim_vals, y_interp, color=c, alpha=0.04, linewidth=0.5)

        if not curves_list:
            continue
        arr  = np.array(curves_list)
        mean = np.nanmean(arr, axis=0)
        ci   = 1.96 * np.nanstd(arr, axis=0) / np.sqrt(len(arr))
        ax2.fill_between(sim_vals, mean - ci, mean + ci, color=c, alpha=0.15)
        ax2.plot(sim_vals, mean, color=c, linewidth=2.0)

    cax2 = ax2.inset_axes([0.0, 1.01, 1.0, 0.04])
    cb2  = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1, 1), cmap=cmap_sent.reversed()),
        cax=cax2, orientation='horizontal',
        ticks=[-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
    )
    cb2.ax.set_xticklabels(['-100%','-66%','-33%','0%','33%','66%','100%'], fontsize=8)
    cb2.ax.xaxis.set_ticks_position('top')
    cb2.ax.set_title("Sentiment", fontsize=9, pad=10)

    ax2.set_xlabel("Similarity")
    ax2.set_ylabel("LooM")
    ax2.set_xlim([sim_vals[0], sim_vals[-1]])
    ax2.set_ylim([-1.05, 1.05])
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax2.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax2.grid(False)

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#333333')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(f"{OUTPUT_DIR}/LooM.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved {OUTPUT_DIR}/LooM.png")


if __name__ == "__main__":
    sent_dict, sim_dict = load_data()
    plot(sent_dict, sim_dict)