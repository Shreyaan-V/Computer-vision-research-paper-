"""Canny, Sobel, and Laplacian edge detection with a reproducible CLI."""

import argparse
from pathlib import Path

import cv2
import numpy as np


METHODS = ("canny", "sobel", "laplacian")


def detect_edges(image, method="canny", *, low=50, high=150, kernel=3, blur=3):
    """Return an 8-bit grayscale edge map from an 8-bit gray/BGR/BGRA image."""
    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("image must be a nonempty 8-bit NumPy array")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("image must be grayscale, BGR, or BGRA")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if kernel not in (1, 3, 5, 7) or blur not in (0, 1, 3, 5, 7, 9, 11):
        raise ValueError("kernel must be 1, 3, 5, or 7; blur must be 0 or an odd value from 1 to 11")
    if not (isinstance(low, int) and isinstance(high, int) and 0 <= low < high <= 255):
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")

    if image.ndim == 2:
        gray = image
    else:
        code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(image, code)
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    if method == "canny":
        return cv2.Canny(gray, low, high, apertureSize=max(kernel, 3), L2gradient=True)
    if method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
        magnitude = np.hypot(gx, gy)
    else:
        magnitude = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel))
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Save a Canny, Sobel, or Laplacian edge map.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="output PNG path")
    parser.add_argument("--method", choices=METHODS, default="canny")
    parser.add_argument("--low", type=int, default=50)
    parser.add_argument("--high", type=int, default=150)
    parser.add_argument("--kernel", type=int, choices=(1, 3, 5, 7), default=3)
    parser.add_argument("--blur", type=int, choices=(0, 1, 3, 5, 7, 9, 11), default=3)
    args = parser.parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
    if image is None:
        parser.error(f"could not read image: {args.input}")
    if args.output.suffix.lower() != ".png":
        parser.error("output filename must end in .png")
    try:
        result = detect_edges(image, args.method, low=args.low, high=args.high,
                              kernel=args.kernel, blur=args.blur)
    except ValueError as exc:
        parser.error(str(exc))
    if not cv2.imwrite(str(args.output), result):
        parser.error(f"could not write output: {args.output}")
    print(f"Saved {args.method} edge map to {args.output} ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    main()
