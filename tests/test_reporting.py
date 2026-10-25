"""Plot ranges must show meaningful differences without clipping uncertainty."""
import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from elliptic_models import ResultPlots


class PlotRangeTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_focuses_on_small_score_range(self):
        _, ax = plt.subplots()
        ResultPlots.relevant_ylim(ax, [.10, .12, .15], [.01, .02, .01])
        low, high = ax.get_ylim()
        self.assertLess(low, .09)
        self.assertGreater(high, .16)
        self.assertLess(high - low, .15)

    def test_constant_scores_still_have_a_visible_range(self):
        _, ax = plt.subplots()
        ResultPlots.relevant_ylim(ax, [.9, .9])
        low, high = ax.get_ylim()
        self.assertLess(low, .9)
        self.assertGreater(high, .9)
        self.assertLess(high - low, .05)

    def test_uncertainty_is_not_clipped_at_metric_bounds(self):
        _, ax = plt.subplots()
        ResultPlots.relevant_ylim(ax, [.02, .98], [.05, .05])
        low, high = ax.get_ylim()
        self.assertLess(low, -.03)
        self.assertGreater(high, 1.03)

    def test_delta_keeps_zero_reference(self):
        _, ax = plt.subplots()
        ResultPlots.relevant_ylim(ax, [.03, .04], include_zero=True, bounded=False)
        low, high = ax.get_ylim()
        self.assertLess(low, 0.)
        self.assertGreater(high, .04)
        self.assertLess(high - low, .1)


if __name__ == '__main__':
    unittest.main()
