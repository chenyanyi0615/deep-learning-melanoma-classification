import unittest
import numpy as np
from scripts.compare_thresholds import select_threshold, threshold_sweep
from src.metrics import binary_metrics


class ThresholdChecks(unittest.TestCase):
    def test_highest_feasible_with_ties(self):
        y = np.array([1, 1, 1, 1, 0, 0])
        p = np.array([.8, .6, .6, .1, .7, .2])
        t = select_threshold(y, p, .75)
        self.assertEqual(t, .6)
        self.assertEqual(binary_metrics(y, p, t)['sensitivity'], .75)
        self.assertLess(binary_metrics(y, p, np.nextafter(t, 1))['sensitivity'], .75)

    def test_all_positives_required(self):
        self.assertEqual(select_threshold([1, 1, 0], [.9, .1, .4], 1), .1)

    def test_sweep_matches_confusion_metrics(self):
        y, p = np.array([0, 1, 0, 1]), np.array([0., .5, .5, 1.])
        for row in threshold_sweep(y, p).itertuples():
            actual = binary_metrics(y, p, row.threshold)
            for key in ['sensitivity', 'specificity', 'precision', 'f1']:
                self.assertAlmostEqual(getattr(row, key), actual[key])

    def test_invalid_inputs_rejected(self):
        for y, p, target in [([0, 0], [.2, .3], .9), ([0, 1], [.1, float('nan')], .9),
                             ([0, 1], [.1, 1.1], .9), ([0, 1], [.1, .2], 0)]:
            with self.assertRaises(ValueError):
                select_threshold(y, p, target)


if __name__ == '__main__':
    unittest.main()
