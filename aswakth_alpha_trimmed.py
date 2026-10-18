"""
ASWAKTH  --  ALPHA-TRIMMED MEAN FILTER
=======================================
Sort the k x k window (mn = k*k samples), delete the d/2 LOWEST and d/2 HIGHEST values,
average the remaining (mn - d) values:

      f_hat(x,y) = 1/(mn - d) * sum of remaining g_r(s,t)

  d = 0       -> arithmetic mean filter
  d = mn - 1  -> median filter
  0 < d < mn-1-> a blend: good for MIXED noise (e.g. Gaussian + salt & pepper)

Contents
  1. alpha_trimmed_naive()  - explicit loops (shows the algorithm)
  2. alpha_trimmed_filter() - fast vectorised version
  3. verification (naive == fast, d=0 == mean, d=mn-1 == median)
  4. experiments: mixed noise, d sweep, comparison with mean & median, window size

Run:  python aswakth_alpha_trimmed.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter as scipy_median, uniform_filter
from common import *


# ----------------------------------------------------------------- filters
def alpha_trimmed_naive(img, k=3, d=2):
    assert d % 2 == 0 and 0 <= d < k * k, "d must be even and < k*k"
    p = k // 2
    padded = np.pad(img, p, mode="reflect")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neigh = np.sort(padded[i:i + k, j:j + k].flatten())
            kept = neigh[d // 2: len(neigh) - d // 2]       # drop d/2 smallest and d/2 largest
            out[i, j] = kept.mean()
    return out


def alpha_trimmed_filter(img, k=3, d=2):
    assert d % 2 == 0 and 0 <= d < k * k, "d must be even and < k*k"
    n = k * k
    w = np.sort(get_windows(img, k), axis=-1)
    return w[..., d // 2: n - d // 2].mean(axis=-1)


def mean_filter(img, k=3):
    return get_windows(img, k).mean(axis=-1)


def median_filter(img, k=3):
    return np.median(get_windows(img, k), axis=-1)


# ----------------------------------------------------------------- verification
def verify(img):
    print("\n[Aswakth] Verification")
    crop = img[100:164, 100:164]
    print("  naive == fast (d=4)        :", np.allclose(alpha_trimmed_naive(crop, 3, 4),
                                                       alpha_trimmed_filter(crop, 3, 4)))
    print("  d=0  == mean filter        :", np.allclose(alpha_trimmed_filter(img, 3, 0),
                                                       uniform_filter(img, 3, mode="mirror")))
    print("  d=8  == median filter (SciPy):", np.allclose(alpha_trimmed_filter(img, 3, 8),
                                                         scipy_median(img, 3, mode="mirror")))


# ----------------------------------------------------------------- experiments
def exp_main(clean, out):
    noisy = add_mixed(clean, sigma=0.06, density=0.08)
    res = {"Mean": mean_filter(noisy), "Median": median_filter(noisy),
           "Alpha-trimmed d=4": alpha_trimmed_filter(noisy, 3, 4)}
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 8))
    show(ax[0, 0], clean, "Original")
    show_m(ax[0, 1], clean, noisy, "Mixed: Gaussian + 8% S&P")
    ax[0, 2].axis("off")
    for a, (n, r) in zip([ax[1, 0], ax[1, 1], ax[1, 2]], res.items()):
        show_m(a, clean, r, n + " 3x3")
    fig.suptitle("Alpha-trimmed mean on mixed noise", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/aswakth_fig1_mixed_noise.png"); plt.close(fig)


def exp_d_visual(clean, out):
    noisy = add_mixed(clean, 0.06, 0.08)
    ds = [0, 2, 4, 6, 8]
    fig, ax = plt.subplots(1, 6, figsize=(20, 3.9))
    show_m(ax[0], clean, noisy, "Noisy")
    for a, d in zip(ax[1:], ds):
        tag = " (=mean)" if d == 0 else " (=median)" if d == 8 else ""
        show_m(a, clean, alpha_trimmed_filter(noisy, 3, d), f"d={d}{tag}")
    fig.suptitle("Effect of the trimming parameter d (3x3 window)", fontsize=13, fontweight="bold")
    fig.savefig(f"{out}/aswakth_fig2_d_visual.png"); plt.close(fig)


def exp_d_curve(clean, out):
    cases = {"Gaussian σ=0.10": add_gaussian(clean, 0.10),
             "Salt & pepper 10%": add_salt_pepper(clean, 0.10),
             "Mixed (G + S&P)": add_mixed(clean, 0.05, 0.05)}
    ds = [0, 2, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for n, noisy in cases.items():
        ax.plot(ds, [psnr(clean, alpha_trimmed_filter(noisy, 3, d)) for d in ds], marker="o", lw=2, label=n)
    ax.set_xlabel("d  (0 = mean … 8 = median)"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Choosing d: depends on the noise type")
    ax.grid(alpha=.3); ax.legend()
    fig.savefig(f"{out}/aswakth_fig3_d_curve.png"); plt.close(fig)


def exp_5x5(clean, out):
    noisy = add_mixed(clean, 0.06, 0.10)
    ds = [0, 4, 8, 12, 16, 24]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    show_m(ax[0], clean, noisy, "Noisy")
    for a, d in zip(ax[1:], [4, 12, 16]):
        show_m(a, clean, alpha_trimmed_filter(noisy, 5, d), f"5x5, d={d}")
    fig.suptitle("Larger window: more samples to trim → handles denser impulses", fontsize=13, fontweight="bold")
    fig.savefig(f"{out}/aswakth_fig4_window_5x5.png"); plt.close(fig)


def exp_noise_density(clean, out):
    dens = np.arange(0.02, 0.32, 0.03)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for n, f in {"Mean": mean_filter, "Median": median_filter,
                 "Alpha d=2": lambda x: alpha_trimmed_filter(x, 3, 2),
                 "Alpha d=4": lambda x: alpha_trimmed_filter(x, 3, 4),
                 "Alpha d=6": lambda x: alpha_trimmed_filter(x, 3, 6)}.items():
        ax.plot(dens * 100, [psnr(clean, f(add_mixed(clean, 0.05, d))) for d in dens], marker="o", label=n)
    ax.set_xlabel("Impulse density (%) on top of Gaussian σ=0.05"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Mixed noise: Alpha-trimmed vs mean vs median")
    ax.grid(alpha=.3); ax.legend()
    fig.savefig(f"{out}/aswakth_fig5_density_sweep.png"); plt.close(fig)


if __name__ == "__main__":
    clean = load_image()
    out = results_dir("aswakth_alpha_trimmed")
    verify(clean)
    exp_main(clean, out)
    exp_d_visual(clean, out)
    exp_d_curve(clean, out)
    exp_5x5(clean, out)
    exp_noise_density(clean, out)
    print(f"\n[Aswakth] Done. Figures in {out}")
