import unittest
import numpy as np
from piglot.utils.assorted import pretty_time, reverse_pretty_time, filter_close_points
from piglot.utils.assorted import stats_interp_to_common_grid


class TestPrettyTime(unittest.TestCase):
    def test_pretty_time(self):
        self.assertEqual(pretty_time(3661), '1h1m1s')
        self.assertEqual(pretty_time(86400), '1d')
        self.assertEqual(pretty_time(0.12345), '0.12345s')


class TestReversePrettyTime(unittest.TestCase):
    def test_reverse_pretty_time(self):
        self.assertEqual(reverse_pretty_time('1h1m1s'), 3661)
        self.assertEqual(reverse_pretty_time('1d'), 86400)
        self.assertEqual(reverse_pretty_time('0.12345s'), 0.12345)


class TestFilterClosePoints(unittest.TestCase):
    def test_filter_close_points(self):
        data = np.array([0, 0.1, 0.1, 0.3, 0.3, 0.3])
        tol = 0.01
        self.assertTrue(np.array_equal(filter_close_points(data, tol), np.array([0, 0.1, 0.3])))


class TestStatsInterpToCommonGrid(unittest.TestCase):
    def test_stats_interp_to_common_grid(self):
        responses = [
            (np.array([0, 0.1, 0.2, 0.3]), np.array([0, 1, 2, 3])),
            (np.array([0.1, 0.2, 0.3]), np.array([2, 3, 4])),
        ]
        result = stats_interp_to_common_grid(responses)
        self.assertTrue(np.array_equal(result['grid'], np.array([0, 0.1, 0.2, 0.3])))
        self.assertTrue(np.array_equal(result['num_points'], np.array([1, 2, 2, 2])))
        self.assertTrue(np.array_equal(result['responses'][0],  np.array([0, 1, 2, 3])))
        self.assertTrue(np.array_equal(result['responses'][1],  np.array([np.nan, 2, 3, 4]),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(result['average'],  np.array([0, 1.5, 2.5, 3.5])))
        self.assertTrue(np.array_equal(result['variance'],  np.array([0, 0.25, 0.25, 0.25])))
        self.assertTrue(np.array_equal(result['std'],  np.array([0, 0.5, 0.5, 0.5])))
        self.assertTrue(np.array_equal(result['min'],  np.array([0, 1, 2, 3])))
        self.assertTrue(np.array_equal(result['max'],  np.array([0, 2, 3, 4])))
        self.assertTrue(np.array_equal(result['median'],  np.array([0, 1.5, 2.5, 3.5])))


if __name__ == '__main__':
    unittest.main()
