import unittest

import numpy as np

from edge_detection import detect_edges


class EdgeDetectionTests(unittest.TestCase):
    def setUp(self):
        self.flat = np.zeros((80, 80), dtype=np.uint8)
        self.step = self.flat.copy()
        self.step[:, 40:] = 255

    def test_flat_image_has_no_edges(self):
        for method in ("canny", "sobel", "laplacian"):
            with self.subTest(method=method):
                result = detect_edges(self.flat, method)
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.shape, self.flat.shape)
                self.assertEqual(np.count_nonzero(result), 0)

    def test_vertical_boundary_detected(self):
        for method in ("canny", "sobel", "laplacian"):
            with self.subTest(method=method):
                result = detect_edges(self.step, method, blur=0)
                self.assertGreater(np.count_nonzero(result[:, 37:43]), 0)
                self.assertEqual(np.count_nonzero(result[:, :30]), 0)
                self.assertEqual(np.count_nonzero(result[:, 50:]), 0)

    def test_color_and_bad_settings(self):
        color = np.stack([self.step] * 3, axis=-1)
        self.assertEqual(detect_edges(color).shape, self.step.shape)
        with self.assertRaises(ValueError):
            detect_edges(color, low=150, high=50)
        with self.assertRaises(ValueError):
            detect_edges(color, blur=4)


if __name__ == "__main__":
    unittest.main()
