# Research and evaluation: edge detection

## Aim

Compare three classical edge detectors on the same image. An edge is a strong local brightness change. It may indicate an object boundary, texture, or a shadow. The tool converts color to grayscale first, so color boundaries with little brightness difference can disappear.

## Methods

| Method | Principle | Strength and limitation |
| --- | --- | --- |
| Sobel | Compute horizontal and vertical derivatives; combine as √(Gx² + Gy²). | Shows gradient strength, but may produce thick or noisy responses. |
| Laplacian | Compute a second spatial derivative and take its absolute value. | Responds in all directions, but is especially noise sensitive and may show two sides of a boundary. |
| Canny | Estimate gradients, thin responses by non-maximum suppression, and connect edges using two thresholds. | Usually yields thin binary edges; results depend strongly on threshold settings. |

Optional Gaussian smoothing reduces noise before detection but can erase small features. Sobel and Laplacian use floating-point calculations to preserve negative derivatives, then clip magnitude into an 8-bit output. This fixed scale makes parameter changes comparable, though strong gradients saturate. Canny uses the L2 gradient setting in OpenCV.

## Reproducible evaluation

Run `python -m unittest discover -s tests -v`. The synthetic 80 × 80 test image contains a vertical black-to-white boundary at column 40. For each method, tests check for a response near the boundary, none far away, and none on a uniform image. This verifies basic behavior without requiring external data. It does not establish accuracy on photographs.

With the default settings, running all three CLI methods on that synthetic image produced these nonzero pixel counts:

| Canny | Sobel | Laplacian |
| ---: | ---: | ---: |
| 80 | 320 | 320 |

Canny's single-pixel response is thinner; the derivative magnitude maps respond over multiple nearby pixels. These counts describe this one constructed boundary, not general accuracy.

For a visual comparison, upload one photograph and switch detectors while keeping blur and kernel fixed. Inspect a low-texture object boundary and a textured region. Record the source, settings, missed boundaries, and spurious texture responses. Canny thresholds control how many candidate edges survive; larger blur generally reduces fine texture. Outcomes depend on resolution, lighting, contrast, and noise.

## Limitations and future work

The tool processes one image at a time and cannot identify semantic objects. Large images may take longer or use substantial memory. A future study could compare output against hand-labeled boundaries using precision, recall, and F1 score, across multiple real-world image types.

## References

1. J. Canny, “A Computational Approach to Edge Detection,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-8, no. 6, pp. 679–698, 1986. [DOI: 10.1109/TPAMI.1986.4767851](https://doi.org/10.1109/TPAMI.1986.4767851).
2. OpenCV, [Sobel Derivatives](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html).
3. OpenCV, [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).
4. OpenCV, [Laplace Operator](https://docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html).
