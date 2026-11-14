import unittest

import numpy as np
import torch

from src.losses import BinaryFocalLoss, make_loss
from src.metrics import binary_metrics


class TrainingChecks(unittest.TestCase):
    def test_metrics_known_confusion(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
        self.assertEqual(result['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(result['auroc'], 0.75)
        for name in ['sensitivity', 'specificity', 'precision', 'f1']:
            self.assertEqual(result[name], 0.5)

    def test_no_positive_predictions(self):
        result = binary_metrics([0, 1], [0.1, 0.2])
        self.assertEqual(result['precision'], 0)
        self.assertEqual(result['f1'], 0)
        self.assertEqual(result['sensitivity'], 0)
        self.assertEqual(result['specificity'], 1)

    def test_focal_gamma_zero(self):
        logits = torch.tensor([-5., 0., 3.], requires_grad=True)
        labels = torch.tensor([0., 1., 1.])
        actual = BinaryFocalLoss(gamma=0, alpha=0.5)(logits, labels)
        expected = .5*torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        torch.testing.assert_close(actual, expected)

    def test_losses_finite_at_extreme_logits(self):
        labels = torch.tensor([1., 0., 1., 0.])
        for name in ['bce', 'weighted_bce', 'focal']:
            logits = torch.tensor([-100., 100., 100., -100.], requires_grad=True)
            loss = make_loss(name, np.array([0, 0, 0, 1]), 'cpu')(logits, labels)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weight_comes_from_training_counts(self):
        loss = make_loss('weighted_bce', np.array([0, 0, 0, 1]), 'cpu')
        self.assertEqual(loss.pos_weight.item(), 3)


if __name__ == '__main__':
    unittest.main()
