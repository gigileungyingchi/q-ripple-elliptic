"""Explicit paired classical/quantum comparisons on fixed experiment contexts."""
import pandas as pd

from .metrics import Evaluation
from .reporting import ResultPlots


class MatchedComparisons:
    """Join by seed/configuration, never by best test result or row position.

    RBF is a matched classical baseline, not a mathematical equivalent of a
    quantum kernel. The caller must supply runs sharing inputs and readout.
    """
    metrics = list(Evaluation.CORE_METRICS) + ['Validation PR-AUC']

    @classmethod
    def pair(cls, results, classical, quantum, keys, label):
        left = results[results.Model == classical]
        right = results[results.Model == quantum]
        if left.empty or right.empty:
            raise ValueError(f'Missing comparison arm: {classical} / {quantum}')
        paired = left[list(keys) + cls.metrics].merge(
            right[list(keys) + cls.metrics], on=list(keys), how='outer',
            suffixes=(' classical', ' quantum'), validate='one_to_one', indicator=True)
        if not paired['_merge'].eq('both').all():
            raise ValueError(f'Unmatched configurations in {label}')
        paired = paired.drop(columns='_merge')
        paired['Comparison'], paired['Classical'], paired['Quantum'] = label, classical, quantum
        for metric in cls.metrics:
            paired[f'Δ {metric}'] = paired[f'{metric} quantum'] - paired[f'{metric} classical']
        return paired

    @classmethod
    def plot(cls, pairs, x='Hops', title='Quantum − matched classical baseline'):
        metrics = [f'Δ {m}' for m in Evaluation.CORE_METRICS]
        fig = ResultPlots.metric_grid(pairs, x=x, series='Comparison', metrics=metrics, title=title)
        for ax, metric in zip(fig.axes, metrics):
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            stats = pairs.groupby(['Comparison', x])[metric].agg(['mean', 'std'])
            ResultPlots.relevant_ylim(ax, stats['mean'], stats['std'].fillna(0),
                                      include_zero=True, bounded=False)
        return fig
