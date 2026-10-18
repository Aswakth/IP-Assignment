"""
main_demo.py  --  COMBINED COMPARISON (run this for the final demo)
Imports each team member's filter (Saiabhyankar, Sri Ramkumar, Aswakth) and compares all of them on every noise type.

Run:  python main_demo.py                    (cameraman test image)
      python main_demo.py my_photo.jpg       (your own image)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from common import *
import saiabhyankar_median as saiabhyankar
import sriramkumar_minmax_midpoint as sriramkumar
import aswakth_alpha_trimmed as aswakth

FILTERS = {
    "Median":        lambda x: saiabhyankar.median_filter(x, 3),
    "Min":           lambda x: sriramkumar.min_filter(x, 3),
    "Max":           lambda x: sriramkumar.max_filter(x, 3),
    "Midpoint":      lambda x: sriramkumar.midpoint_filter(x, 3),
    "Alpha-trimmed": lambda x: aswakth.alpha_trimmed_filter(x, 3, 4),
}


def grid(clean, out):
    noises = {
        "Salt & pepper 10%": add_salt_pepper(clean, 0.10),
        "Gaussian σ=0.10":   add_gaussian(clean, 0.10),
        "Uniform ±0.30":     add_uniform(clean, 0.30),
        "Mixed (G + S&P)":   add_mixed(clean, 0.05, 0.05),
    }
    cols = ["Noisy"] + list(FILTERS)
    fig, ax = plt.subplots(len(noises), len(cols), figsize=(2.7 * len(cols), 3 * len(noises)))
    table = {}
    for i, (nn, noisy) in enumerate(noises.items()):
        table[nn] = {"Noisy": psnr(clean, noisy)}
        show(ax[i, 0], noisy, f"{nn}\n{table[nn]['Noisy']:.1f} dB")
        for j, (fn, f) in enumerate(FILTERS.items(), start=1):
            r = f(noisy)
            table[nn][fn] = psnr(clean, r)
            show(ax[i, j], r, f"{fn}\n{table[nn][fn]:.1f} dB / SSIM {ssim(clean, r):.2f}")
    fig.suptitle("Which filter for which noise?", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/final_fig1_comparison_grid.png"); plt.close(fig)
    return table


def heatmap(table, out):
    rows, cols = list(table), list(next(iter(table.values())))
    M = np.array([[table[r][c] for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontweight="bold")
        ax.add_patch(plt.Rectangle((np.argmax(M[i]) - .5, i - .5), 1, 1, fill=False, lw=3, ec="navy"))
    ax.set_title("PSNR (dB) - navy box = best for that noise")
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.savefig(f"{out}/final_fig2_psnr_heatmap.png"); plt.close(fig)


def density_curve(clean, out):
    dens = np.arange(0.05, 0.55, 0.05)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for n, f in FILTERS.items():
        ax.plot(dens * 100, [psnr(clean, f(add_salt_pepper(clean, d))) for d in dens], marker="o", lw=2, label=n)
    ax.set_xlabel("Salt & pepper density (%)"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Robustness to impulse noise (3x3)")
    ax.grid(alpha=.3); ax.legend()
    fig.savefig(f"{out}/final_fig3_density_curve.png"); plt.close(fig)


def save_csv(table, out):
    cols = list(next(iter(table.values())))
    with open(f"{out}/final_psnr_table.csv", "w", encoding="utf-8-sig") as f:
        f.write("Noise," + ",".join(cols) + "\n")
        for r, v in table.items():
            f.write(r + "," + ",".join(f"{v[c]:.2f}" for c in cols) + "\n")


if __name__ == "__main__":
    clean = load_image(sys.argv[1] if len(sys.argv) > 1 else None)
    out = results_dir("final_comparison")
    table = grid(clean, out)
    heatmap(table, out)
    density_curve(clean, out)
    save_csv(table, out)
    print(f"{'':20s}" + "".join(f"{c:>15s}" for c in next(iter(table.values()))))
    for r, v in table.items():
        print(f"{r:20s}" + "".join(f"{x:15.2f}" for x in v.values()))
    print(f"\nDone. Figures in {out}")