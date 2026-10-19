"""
denoise.py  --  apply our order-statistic filters to YOUR OWN noisy images.

Put noisy images in the  test_images/  folder (or any folder); results are saved in  outputs/ .

Usage examples
    python denoise.py                                    # every image in test_images/, median 3x3
    python denoise.py test_images --filter alpha --k 5 --d 12
    python denoise.py test_images/1_cameraman_saltpepper_10.png --filter median   # one file
    python denoise.py my_noisy_photos --filter midpoint --out my_results

Filters : median | min | max | midpoint | alpha
Output  : <out>/restored/<name>_<filter>.png          restored image
          <out>/comparison/<name>_<filter>_compare.png noisy (left) vs restored (right)
Colour images are filtered channel by channel.
"""
import argparse
import os
import cv2
import numpy as np

import saiabhyankar_median as sai
import sriramkumar_minmax_midpoint as sri
import aswakth_alpha_trimmed as asw

FILTERS = {
    "median":   lambda im, k, d: sai.median_filter(im, k),
    "min":      lambda im, k, d: sri.min_filter(im, k),
    "max":      lambda im, k, d: sri.max_filter(im, k),
    "midpoint": lambda im, k, d: sri.midpoint_filter(im, k),
    "alpha":    lambda im, k, d: asw.alpha_trimmed_filter(im, k, d),
}


EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_images(path):
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if f.lower().endswith(EXTS) and not f.lower().startswith("original"))
        return files
    return [path]


def process(path, a, d, restored_dir, compare_dir):
    flag = cv2.IMREAD_GRAYSCALE if a.gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        print(f"  skipped (cannot read): {path}")
        return
    x = img.astype(np.float64) / 255.0                 # scale to [0, 1]
    f = FILTERS[a.filter]
    if x.ndim == 2:
        y = f(x, a.k, d)
    else:
        y = np.dstack([f(x[..., c], a.k, d) for c in range(x.shape[2])])
    out = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
    base = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(restored_dir, f"{base}_{a.filter}.png"), out)
    cv2.imwrite(os.path.join(compare_dir, f"{base}_{a.filter}_compare.png"), np.hstack([img, out]))
    print(f"  done: {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(description="Order-statistic noise reduction")
    ap.add_argument("input", nargs="?", default="test_images",
                    help="an image file or a folder of images (default: test_images)")
    ap.add_argument("--filter", choices=FILTERS, default="median")
    ap.add_argument("--k", type=int, default=3, help="window size (odd), default 3")
    ap.add_argument("--d", type=int, default=None,
                    help="alpha-trimmed only: samples trimmed (even, < k*k). default = 2*(k*k//4)")
    ap.add_argument("--gray", action="store_true", help="convert to grayscale first")
    ap.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    a = ap.parse_args()

    if a.k % 2 == 0 or a.k < 3:
        ap.error("--k must be an odd number >= 3")
    d = a.d if a.d is not None else 2 * (a.k * a.k // 4)
    if a.filter == "alpha" and (d % 2 or not 0 <= d < a.k * a.k):
        ap.error(f"--d must be even and < k*k ({a.k * a.k}); got {d}")

    files = find_images(a.input)
    if not files or not os.path.exists(files[0]):
        raise SystemExit(f"No images found at: {a.input}")

    restored_dir = os.path.join(a.out, "restored")
    compare_dir = os.path.join(a.out, "comparison")
    os.makedirs(restored_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    extra = f", d={d}" if a.filter == "alpha" else ""
    print(f"Filter: {a.filter} (window {a.k}x{a.k}{extra}) on {len(files)} image(s)")
    for p in files:
        process(p, a, d, restored_dir, compare_dir)
    print(f"\nRestored images : {restored_dir}")
    print(f"Comparisons     : {compare_dir}  (left = noisy, right = restored)")


if __name__ == "__main__":
    main()