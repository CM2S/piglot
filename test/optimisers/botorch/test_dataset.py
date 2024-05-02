import unittest
import torch
from piglot.optimisers.botorch.dataset import BayesDataset


class TestBayesDataset(unittest.TestCase):
    def setUp(self):
        self.repeats = 1
        self.n_outs = 20
        self.n_points = 50
        self.dtype = torch.float64
        self.params = torch.rand(self.n_points, 6, dtype=self.dtype).repeat(self.repeats, 1)
        self.values = torch.rand(self.n_points, self.n_outs, dtype=self.dtype).repeat(self.repeats, 2)
        self.variances = torch.rand(self.n_points, self.n_outs, dtype=self.dtype).repeat(self.repeats, 2)

    def test_transform_normal(self):
        dataset = BayesDataset(6, 2 * self.n_outs)
        for i in range(self.n_points * self.repeats):
            dataset.push(self.params[i, :], self.values[i, :], self.variances[i, :])
        values, variances = dataset.transform_outcomes(self.values, self.variances)
        untransform = dataset.untransform_outcomes(values)
        diff = torch.abs(untransform - self.values)
        self.assertTrue(torch.all(diff < 1e-6))

    def test_transform_pca(self):
        dataset = BayesDataset(6, 2 * self.n_outs, pca_variance=0.999)
        for i in range(self.n_points * self.repeats):
            dataset.push(self.params[i, :], self.values[i, :], self.variances[i, :])
        values, variances = dataset.transform_outcomes(self.values, self.variances)
        untransform = dataset.untransform_outcomes(values)
        diff = torch.abs(untransform - self.values)
        print(torch.mean(diff))
        self.assertTrue(torch.norm(diff) < 1e-6)
