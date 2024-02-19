import unittest
import numpy as np
from piglot.utils.responses import Transformer, reduce_response, interpolate_response


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = Transformer(
            x_scale=2.0,
            y_scale=3.0,
            x_offset=1.0,
            y_offset=2.0,
            x_min=0,
            x_max=10,
        )

    def test_call(self):
        x_old = np.array([0, 1, 2, 3, 4, 5])
        y_old = np.array([0, 1, 2, 3, 4, 5])
        x_new, y_new = self.transformer(x_old, y_old)
        self.assertTrue(np.array_equal(x_new, np.array([1, 3, 5, 7, 9])))
        self.assertTrue(np.array_equal(y_new, np.array([2, 5, 8, 11, 14])))

    def test_read(self):
        config = {
            'x_scale': 2.0,
            'y_scale': 3.0,
            'x_offset': 1.0,
            'y_offset': 2.0,
            'x_min': 0,
            'x_max': 10
        }
        transformer = Transformer.read(config)
        self.assertEqual(transformer.x_scale, 2.0)
        self.assertEqual(transformer.y_scale, 3.0)
        self.assertEqual(transformer.x_offset, 1.0)
        self.assertEqual(transformer.y_offset, 2.0)
        self.assertEqual(transformer.x_min, 0)
        self.assertEqual(transformer.x_max, 10)


class TestReduceResponse(unittest.TestCase):
    def test_reduce_response(self):
        x_old = np.array([0, 1, 2, 3, 4, 5])
        y_old = np.array([0, 1, 2, 2, 2, 2])
        tol = 0.1
        n_points, error, (x_new, y_new) = reduce_response(x_old, y_old, tol)
        self.assertEqual(n_points, 3)
        self.assertAlmostEqual(error, 0.0, places=1)
        self.assertTrue(np.array_equal(x_new, np.array([0, 2, 5])))
        self.assertTrue(np.array_equal(y_new, np.array([0, 2, 2])))

    def test_reduce_response_tol(self):
        x_old = np.array([0, 1, 2, 3, 4, 5])
        y_old = np.array([0, 1, 2, 1, 2, 3])
        tol = 1e-8
        n_points, error, (x_new, y_new) = reduce_response(x_old, y_old, tol)
        self.assertEqual(n_points, 4)
        self.assertAlmostEqual(error, 0.0, places=1)
        self.assertTrue(np.array_equal(x_new, np.array([0, 2, 3, 5])))
        self.assertTrue(np.array_equal(y_new, np.array([0, 2, 1, 3])))


class TestInterpolateResponse(unittest.TestCase):
    def test_interpolate_response(self):
        x_resp = np.array([0, 2, 4])
        y_resp = np.array([0, 2, 4])
        x_grid = np.array([0, 1, 2, 3, 4, 5])
        x_zeros = np.array([0])
        x_zeros2 = np.array([0, 0])
        self.assertTrue(np.array_equal(interpolate_response(x_resp, y_resp, x_grid),
                                       np.array([0, 1, 2, 3, 4, 4])))
        self.assertTrue(np.array_equal(interpolate_response(x_zeros, x_zeros, x_grid),
                                       np.zeros_like(x_grid)))
        self.assertTrue(np.array_equal(interpolate_response(x_zeros2, x_zeros2, x_grid),
                                       np.zeros_like(x_grid)))


if __name__ == '__main__':
    unittest.main()
