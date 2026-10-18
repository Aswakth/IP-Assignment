"""
SAIABHYANKAR  --  MEDIAN FILTER
===========================
Idea: replace each pixel by the MEDIAN of its k x k neighbourhood.
      f_hat(x,y) = median{ g(s,t) : (s,t) in S_xy }
Best for: salt & pepper (impulse) noise. Edges are preserved far better than a mean filter.

Contents
  1. median_filter_naive()  - textbook version with explicit loops (shows the algorithm)
  2. median_filter()        - fast vectorised version (used for experiments)
  3. verification           - naive == fast == SciPy
  4. experiments            - impulse removal, window-size study, density sweep, median vs mean

Run:  python saiabhyankar_median.py
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter as scipy_median
from common import *


# ----------------------------------------------------------------- filters
def median_filter_naive(img, k=3):
    """Slow, readable version: visit each pixel, collect neighbours, sort, take middle."""
    p = k // 2
    padded = np.pad(img, p, mode="reflect")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neigh = padded[i:i + k, j:j + k].flatten()
            neigh.sort()
            out[i, j] = neigh[len(neigh) // 2]
    return out


def median_filter(img, k=3):
    """Fast version: build all windows at once and take the median along the last axis."""
    return np.median(get_windows(img, k), axis=-1)


def mean_filter(img, k=3):
    """Baseline only, to show why median beats averaging on impulse noise."""
    return get_windows(img, k).mean(axis=-1)


# ----------------------------------------------------------------- verification
def verify(img):
    print("\n[Saiabhyankar] Verification")
    crop = img[100:164, 100:164]                       # small crop so the naive loop is quick
    t0 = time.time(); a = median_filter_naive(crop, 3); t_naive = time.time() - t0
    t0 = time.time(); b = median_filter(crop, 3);       t_fast = time.time() - t0
    c = scipy_median(crop, 3, mode="mirror")
    print(f"  naive == fast  : {np.allclose(a, b)}")
    print(f"  fast  == SciPy : {np.allclose(b, c)}")
    print(f"  time on 64x64 crop -> naive {t_naive*1000:.1f} ms, vectorised {t_fast*1000:.2f} ms")


# ----------------------------------------------------------------- experiments
def exp_main(clean, out):
    noisy = add_salt_pepper(clean, 0.15)
    med = median_filter(noisy, 3)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    show(ax[0], clean, "Original")
    show_m(ax[1], clean, noisy, "Salt & pepper 15%")
    show_m(ax[2], clean, med, "Median 3x3")
    show_error(ax[3], clean, med, "|Error| after median")
    fig.suptitle("Median filter: salt-and-pepper restoration", fontsize=13, fontweight="bold")
    fig.savefig(f"{out}/saiabhyankar_fig1_main.png"); plt.close(fig)


def exp_vs_mean(clean, out):
    noisy = add_salt_pepper(clean, 0.15)
    ys, xs = slice(50, 130), slice(150, 230)
    items = {"Noisy": noisy, "Mean 3x3": mean_filter(noisy, 3), "Median 3x3": median_filter(noisy, 3)}
    fig, ax = plt.subplots(2, 3, figsize=(10, 7))
    for j, (n, im) in enumerate(items.items()):
        show_m(ax[0, j], clean, im, n)
        show(ax[1, j], im[ys, xs], f"{n} (zoom)")
    fig.suptitle("Why median? It removes impulses instead of smearing them", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/saiabhyankar_fig2_median_vs_mean.png"); plt.close(fig)


def exp_window(clean, out):
    noisy = add_salt_pepper(clean, 0.25)
    ks = [3, 5, 7, 9]
    fig, ax = plt.subplots(1, 5, figsize=(17, 4))
    show_m(ax[0], clean, noisy, "Noisy 25%")
    for a, k in zip(ax[1:], ks):
        show_m(a, clean, median_filter(noisy, k), f"Median {k}x{k}")
    fig.suptitle("Window size: larger removes more noise but blurs fine detail", fontsize=13, fontweight="bold")
    fig.savefig(f"{out}/saiabhyankar_fig3_window_size.png"); plt.close(fig)


def exp_density(clean, out):
    dens = np.arange(0.05, 0.75, 0.05)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k in (3, 5, 7):
        ax.plot(dens * 100, [psnr(clean, median_filter(add_salt_pepper(clean, d), k)) for d in dens],
                marker="o", lw=2, label=f"Median {k}x{k}")
    ax.plot(dens * 100, [psnr(clean, mean_filter(add_salt_pepper(clean, d), 3)) for d in dens],
            "k--", marker="s", label="Mean 3x3 (baseline)")
    ax.set_xlabel("Salt & pepper density (%)"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Median filter robustness vs noise density")
    ax.grid(alpha=.3); ax.legend()
    fig.savefig(f"{out}/saiabhyankar_fig4_density_sweep.png"); plt.close(fig)


def exp_iterative(clean, out):
    """Repeated median filtering removes the leftover impulses at high noise density."""
    noisy = add_salt_pepper(clean, 0.4)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    show_m(ax[0], clean, noisy, "Noisy 40%")
    cur = noisy
    for i, a in enumerate(ax[1:], start=1):
        cur = median_filter(cur, 3)
        show_m(a, clean, cur, f"Median 3x3, pass {i}")
    fig.suptitle("Iterated median filtering on heavy noise", fontsize=13, fontweight="bold")
    fig.savefig(f"{out}/saiabhyankar_fig5_iterative.png"); plt.close(fig)


if __name__ == "__main__":
    clean = load_image()
    out = results_dir("saiabhyankar_median")
    verify(clean)
    exp_main(clean, out)
    exp_vs_mean(clean, out)
    exp_window(clean, out)
    exp_density(clean, out)
    exp_iterative(clean, out)
    print(f"\n[Saiabhyankar] Done. Figures in {out}")
