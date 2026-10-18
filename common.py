"""
common.py  --  shared utilities used by ALL three team members (Saiabhyankar, Sri Ramkumar, Aswakth).
Contains: test image loader, noise models, window extraction, metrics, plot helpers.
(No filter lives here - each member writes their own filter in their own file.)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # save figures to files (no pop-up window)
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from skimage import data, img_as_float
from skimage.metrics import structural_similarity

# Windows consoles default to cp1252 and cannot print symbols like sigma; force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RNG = np.random.default_rng(42)             # fixed seed -> same results on every run

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.dpi": 130, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def results_dir(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", name)
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------ image
def load_image(path=None):
    """Grayscale float image in [0,1]. Default = 'cameraman'. Pass a path to use your own."""
    if path is None:
        return img_as_float(data.camera())
    import cv2
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im.astype(np.float64) / 255.0


# ------------------------------------------------------------------ noise
def add_salt_pepper(img, density=0.1, salt_ratio=0.5):
    out = img.copy()
    r = RNG.random(img.shape)
    out[r < density * salt_ratio] = 1.0
    out[(r >= density * salt_ratio) & (r < density)] = 0.0
    return out


def add_gaussian(img, sigma=0.1):
    return np.clip(img + RNG.normal(0, sigma, img.shape), 0, 1)


def add_uniform(img, width=0.3):
    return np.clip(img + RNG.uniform(-width, width, img.shape), 0, 1)


def add_mixed(img, sigma=0.05, density=0.05):
    return add_salt_pepper(add_gaussian(img, sigma), density)


# ------------------------------------------------------------------ neighbourhoods
def get_windows(img, k=3):
    """Return array of shape (H, W, k*k): the k x k neighbourhood of every pixel
    (borders handled by reflect padding)."""
    p = k // 2
    padded = np.pad(img, p, mode="reflect")
    win = sliding_window_view(padded, (k, k))
    return win.reshape(img.shape[0], img.shape[1], k * k)


# ------------------------------------------------------------------ metrics
def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(1.0 / m)


def ssim(a, b):
    return float(structural_similarity(a, b, data_range=1.0))


# ------------------------------------------------------------------ plotting
def show(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def show_m(ax, clean, img, title):
    """Show image with PSNR / SSIM in the title."""
    show(ax, img, f"{title}\nPSNR {psnr(clean, img):.2f} dB | SSIM {ssim(clean, img):.3f}")


def show_error(ax, clean, img, title="|Error|"):
    im = ax.imshow(np.abs(clean - img), cmap="inferno", vmin=0, vmax=0.5)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)