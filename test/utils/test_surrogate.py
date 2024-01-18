import unittest
import numpy as np
import torch
from botorch.models import SingleTaskGP
from piglot.utils.surrogate import get_model


class TestGetGPModel(unittest.TestCase):
    def setUp(self):
        self.x_data = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_data = np.array([7, 8, 9])
        self.var_data = np.array([0.1, 0.2, 0.3])

    def test_get_model_noisy(self):
        model = get_model(self.x_data, self.y_data, noisy=True)
        self.assertIsInstance(model, SingleTaskGP)
        self.assertEqual(model.train_inputs[0].shape, torch.Size([3, 2]))
        self.assertEqual(model.train_targets.shape, torch.Size([3]))

    def test_get_model_fixed_noise_no_var_data(self):
        model = get_model(self.x_data, self.y_data)
        self.assertIsInstance(model, SingleTaskGP)
        self.assertEqual(model.train_inputs[0].shape, torch.Size([3, 2]))
        self.assertEqual(model.train_targets.shape, torch.Size([3]))
        self.assertTrue(torch.equal(model.likelihood.noise,
                                    torch.ones_like(model.likelihood.noise) * 1e-6))

    def test_get_model_fixed_noise_with_var_data(self):
        model = get_model(self.x_data, self.y_data, self.var_data)
        self.assertIsInstance(model, SingleTaskGP)
        self.assertEqual(model.train_inputs[0].shape, torch.Size([3, 2]))
        self.assertEqual(model.train_targets.shape, torch.Size([3]))
        self.assertTrue(torch.equal(model.likelihood.noise, torch.tensor(self.var_data)))


if __name__ == '__main__':
    unittest.main()
