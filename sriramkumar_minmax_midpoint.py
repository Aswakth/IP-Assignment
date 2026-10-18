"""
SRI RAMKUMAR  --  MIN, MAX (MIN-MAX) AND MIDPOINT FILTERS
=====================================================
  Min filter      : f_hat = min{ g(s,t) }            -> removes SALT (white) impulses, darkens image
  Max filter      : f_hat = max{ g(s,t) }            -> removes PEPPER (black) impulses, brightens image
  Midpoint filter : f_hat = (max + min) / 2          -> works best for randomly distributed,
                                                        short-tailed noise (Gaussian / uniform)

Contents
  1. min_filter(), max_filter(), midpoint_filter()
  2. verification vs SciPy
  3. experiments: unipolar noise, min/max side effects, midpoint on Gaussian/uniform, min-then-max combos

Run:  python sriramkumar_minmax_midpoint.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter, maximum_filter
from common import *


# ----------------------------------------------------------------- filters
def min_filter(img, k=3):
    return get_windows(img, k).min(axis=-1)


def max_filter(img, k=3):
    return get_windows(img, k).max(axis=-1)


def midpoint_filter(img, k=3):
    w = get_windows(img, k)
    return 0.5 * (w.max(axis=-1) + w.min(axis=-1))


def open_then_close(img, k=3):
    """Extra: min followed by max (morphological opening) - kills salt without darkening much."""
    return max_filter(min_filter(img, k), k)


def close_then_open(img, k=3):
    """Extra: max followed by min (morphological closing) - kills pepper without brightening much."""
    return min_filter(max_filter(img, k), k)


# ----------------------------------------------------------------- verification
def verify(img):
    print("\n[Sri Ramkumar] Verification vs SciPy")
    print("  min      :", np.allclose(min_filter(img), minimum_filter(img, 3, mode="mirror")))
    print("  max      :", np.allclose(max_filter(img), maximum_filter(img, 3, mode="mirror")))
    mid_ref = 0.5 * (minimum_filter(img, 3, mode="mirror") + maximum_filter(img, 3, mode="mirror"))
    print("  midpoint :", np.allclose(midpoint_filter(img), mid_ref))


# ----------------------------------------------------------------- experiments
def exp_unipolar(clean, out):
    salt = add_salt_pepper(clean, 0.10, salt_ratio=1.0)
    pepper = add_salt_pepper(clean, 0.10, salt_ratio=0.0)
    fig, ax = plt.subplots(2, 4, figsize=(15, 8))
    show_m(ax[0, 0], clean, salt, "Salt-only noise")
    show_m(ax[0, 1], clean, min_filter(salt), "Min filter")
    show_m(ax[0, 2], clean, max_filter(salt), "Max filter")
    show_m(ax[0, 3], clean, midpoint_filter(salt), "Midpoint filter")
    show_m(ax[1, 0], clean, pepper, "Pepper-only noise")
    show_m(ax[1, 1], clean, min_filter(pepper), "Min filter")
    show_m(ax[1, 2], clean, max_filter(pepper), "Max filter")
    show_m(ax[1, 3], clean, midpoint_filter(pepper), "Midpoint filter")
    fig.suptitle("Unipolar impulse noise: min removes salt, max removes pepper", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/sriramkumar_fig1_unipolar_noise.png"); plt.close(fig)


def exp_bipolar(clean, out):
    noisy = add_salt_pepper(clean, 0.10)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    show_m(ax[0], clean, noisy, "Salt & pepper 10%")
    show_m(ax[1], clean, min_filter(noisy), "Min")
    show_m(ax[2], clean, max_filter(noisy), "Max")
    show_m(ax[3], clean, midpoint_filter(noisy), "Midpoint")
    fig.suptitle("Bipolar impulse noise: none of them is a good choice alone", fontsize=13, fontweight="bold")
    fig.savefig(f"{out}/sriramkumar_fig2_bipolar_noise.png"); plt.close(fig)


def exp_midpoint_random_noise(clean, out):
    cases = {"Gaussian σ=0.10": add_gaussian(clean, 0.10), "Uniform ±0.30": add_uniform(clean, 0.30)}
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.5))
    for i, (n, noisy) in enumerate(cases.items()):
        show_m(ax[i, 0], clean, noisy, n)
        show_m(ax[i, 1], clean, midpoint_filter(noisy), "Midpoint 3x3")
        show_error(ax[i, 2], clean, midpoint_filter(noisy), "|Error|")
    fig.suptitle("Midpoint filter on random (short-tailed) noise", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/sriramkumar_fig3_midpoint_random_noise.png"); plt.close(fig)


def exp_combo(clean, out):
    """Chaining min and max (opening / closing) to handle salt-and-pepper."""
    noisy = add_salt_pepper(clean, 0.05)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    show_m(ax[0], clean, noisy, "Salt & pepper 5%")
    show_m(ax[1], clean, open_then_close(noisy), "Min→Max (removes salt)")
    show_m(ax[2], clean, close_then_open(noisy), "Max→Min (removes pepper)")
    show_m(ax[3], clean, close_then_open(open_then_close(noisy)), "Open then close")
    fig.suptitle("Extra: combining min and max filters", fontsize=13, fontweight="bold")
    fig.savefig(f"{out}/sriramkumar_fig4_min_max_combination.png"); plt.close(fig)


def exp_window(clean, out):
    noisy = add_gaussian(clean, 0.08)
    ks = [3, 5, 7]
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    show_m(ax[0, 0], clean, noisy, "Gaussian σ=0.08"); ax[1, 0].axis("off")
    for j, k in enumerate(ks, start=1):
        show_m(ax[0, j], clean, midpoint_filter(noisy, k), f"Midpoint {k}x{k}")
        show_m(ax[1, j], clean, max_filter(noisy, k), f"Max {k}x{k}")
    fig.suptitle("Window-size effect", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/sriramkumar_fig5_window_size.png"); plt.close(fig)


if __name__ == "__main__":
    clean = load_image()
    out = results_dir("sriramkumar_minmax_midpoint")
    verify(clean)
    exp_unipolar(clean, out)
    exp_bipolar(clean, out)
    exp_midpoint_random_noise(clean, out)
    exp_combo(clean, out)
    exp_window(clean, out)
    print(f"\n[Sri Ramkumar] Done. Figures in {out}")
