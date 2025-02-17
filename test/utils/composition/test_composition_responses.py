import unittest
import numpy as np
from piglot.utils.composition.responses import (
    FixedFlatteningUtility,
    EndpointFlattenUtility,
    ConcatUtility,
)


class TestFlattenUtilities(unittest.TestCase):
    def test_fixed_flat(self):
        x_ref = np.array([0.0, 0.5, 1.0])
        y_ref = np.array([0.0, 0.5, 1.0])
        utility = FixedFlatteningUtility(x_ref)
        self.assertEqual(utility.length(), 3)
        x_new = np.array([-1.0, 0.5, 2.0])
        y_new = np.array([-1.0, 0.5, 2.0])
        flat = utility.flatten(x_new, y_new)
        self.assertTrue(np.allclose(flat, y_new))
        y_test = np.array([0.5, 1.0, 1.5])
        new_time, new_data = utility.unflatten(y_test)
        self.assertTrue(np.allclose(new_time, x_ref))
        self.assertTrue(np.allclose(new_data, y_test))
        # Assertions
        x_bad = np.array([0.0, 0.5, 1.0, 1.5])
        y_bad = np.array([0.0, 0.5, 1.0, 1.5])
        self.assertRaises(ValueError, utility.flatten, x_bad, y_new)
        self.assertRaises(ValueError, utility.flatten, x_ref, y_bad)

    def test_endpoint_flat(self):
        n_points = 10
        utility = EndpointFlattenUtility(n_points)
        self.assertEqual(utility.length(), n_points + 2)
        x_new = np.array([0.0, 1.0])
        y_new = 2 * x_new
        x_reb, y_reb = utility.unflatten(utility.flatten(x_new, y_new))
        self.assertAlmostEqual(np.trapz(y_reb, x_reb), np.trapz(y_new, x_new))
        # Assertions
        x_bad = np.array([0.0, 0.5, 1.0, 1.5])
        self.assertRaises(ValueError, utility.flatten, x_bad, y_new)
        self.assertRaises(NotImplementedError, utility.flatten_torch, x_new, y_new)


class TestConcatUtility(unittest.TestCase):
    def test_concat_utility(self):
        x1 = np.array([0.0, 1.0])
        x2 = np.array([0.0, 0.5, 1.0])
        y1 = 1 + 2 * x1
        y2 = 2 + 3 * x2
        yvar1 = np.diag(np.ones_like(x1))
        yvar2 = np.diag(np.ones_like(x2))
        utility = ConcatUtility([len(x1), len(x2)])
        values = utility.concat([y1, y2])
        covariances = utility.concat_covar([yvar1, yvar2])
        self.assertTrue(np.allclose(values, np.concatenate([y1, y2])))
        self.assertTrue(np.allclose(covariances, np.diag(np.ones_like(values))))
        y1_rec, y2_rec = utility.split(values)
        self.assertTrue(np.allclose(y1_rec, y1))
        self.assertTrue(np.allclose(y2_rec, y2))


if __name__ == '__main__':
    unittest.main()
