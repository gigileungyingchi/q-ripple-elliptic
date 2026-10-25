"""Evidence-led notebook figures and conclusions, loaded from saved experiments."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import Evaluation
from .reporting import ResultReport, ResultPlots, ExperimentSummary


class TopologyResearchPlots:
    @staticmethod
    def probes(results):
        case = 'Degree + triangle collision (8 nodes)'
        frame = results[results.Case == case]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(frame))
        names = frame.Encoder.replace({
            'Classical summary RBF':'Summary RBF', 'Classical spectral RBF':'Spectral RBF',
            'Classical rooted WL':'Rooted WL', 'Quantum summary rotations':'Q summary',
            'Quantum graph message':'Q graph message', 'Quantum generic ring':'Q generic ring',
            'Quantum Hamiltonian':'Q Hamiltonian fidelity', 'Quantum Hermitian walk':'Q walk fidelity',
            'Quantum Hamiltonian invariant moments (exact)':'Q moments (exact)',
            'Quantum Hamiltonian invariant moments (product)':'Q moments (product)',
            'Quantum Hamiltonian mixture (3)':'Q mixture (3)',
        })
        axes[0].plot(x, frame['Isomorphic mean similarity'], 'o', label='Relabeled same graph (mean)')
        axes[0].plot(x, frame['Different graph similarity'], 's', label='Different graph')
        axes[0].set(ylabel='Kernel similarity', title='Same summary, different connectivity')
        ResultPlots.relevant_ylim(axes[0], np.r_[frame['Isomorphic mean similarity'], frame['Different graph similarity']])
        axes[0].legend(fontsize=8)
        axes[1].plot(x, frame['Mean relabeling error'], 'o', color='tab:red')
        axes[1].set(ylabel='1 − mean isomorphic similarity', title='Relabeling error: lower is better')
        ResultPlots.relevant_ylim(axes[1], frame['Mean relabeling error'], include_zero=True)
        for ax in axes:
            ax.set_xticks(x, names, rotation=45, ha='right', fontsize=8)
            ax.grid(alpha=.2)
        fig.tight_layout()
        return fig

    @staticmethod
    def capacity(retention):
        frame = retention[retention.Split == 'test']
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, column, title in zip(axes,
            ['Retention fraction', 'Edge retention fraction', 'Maximum retained distance'],
            ['Reachable nodes retained', 'Reachable edges retained', 'Actual retained reach']):
            values = []
            for hops, rows in frame.groupby('Hops'):
                means = rows.groupby('Node budget')[column].mean()
                ax.plot(means.index, means, 'o-', label=f'{hops} hops')
                values.extend(means)
            ax.set(xlabel='Node budget', ylabel=column, title=title)
            ResultPlots.relevant_ylim(ax, values, bounded=column != 'Maximum retained distance')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        return fig

    @staticmethod
    def architectures(results, features='Local'):
        frame = results[(results.Axis == 'Architecture') & ~results.Diagnostic &
                        (results['Original features'] == features)]
        models = frame.Model.str.replace('Raw + ', '', regex=False)
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, .3*len(frame))))
        for ax, metric in zip(axes, ['PR-AUC', 'Validation PR-AUC']):
            ax.plot(frame[metric], np.arange(len(frame)), 'o')
            ax.set_yticks(np.arange(len(frame)), models, fontsize=8)
            ax.set(xlabel=metric, title=f'{features}: {metric}')
            low, high = frame[metric].min(), frame[metric].max()
            pad = max((high-low)*.1, .005)
            ax.set_xlim(max(0, low-pad), min(1, high+pad))
            ax.grid(alpha=.2)
        fig.tight_layout()
        return fig


class StorySummary:
    """Keep structural claims separate from illicit-classification comparisons."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.previous = ExperimentSummary(directory)
        self.encodings = pd.read_csv(self.directory / 'topology_encodings.csv')
        self.probes = pd.read_csv(self.directory / 'structural_probes.csv')
        self.retention = pd.read_csv(self.directory / 'encoding_retention.csv')
        self.pairs = pd.read_csv(self.directory / 'matched_encoding_pairs.csv')

    def table(self):
        rows = self.previous.comparison_winners()
        extra = []
        primary = self.encodings[(~self.encodings.Diagnostic) & (self.encodings.Axis == 'Architecture')]
        for features, frame in primary.groupby('Original features', sort=False):
            quantum = frame.Model.str.contains('quantum', case=False)
            for family, candidates in [('Classical', frame[~quantum]), ('Quantum', frame[quantum])]:
                winner = candidates.loc[candidates['Validation PR-AUC'].idxmax()]
                extra.append({'Study': 'New encoders (smaller pilot)', 'Features': features+(' 93' if features == 'Local' else ' 165'),
                              'Family': family, **winner.to_dict()})
            # Keep Hamiltonian, invariant moments AND walk visible even if a different encoder wins.
            for name in ('Raw + Hamiltonian quantum', 'Raw + Hamiltonian moment quantum', 'Raw + Hermitian walk quantum'):
                winner = frame[frame.Model == name].iloc[0]
                extra.append({'Study': 'New encoder detail (smaller pilot)',
                    'Features': features+(' 93' if features == 'Local' else ' 165'),
                    'Family': 'Quantum', **winner.to_dict()})
        result = pd.concat([rows, pd.DataFrame(extra)], ignore_index=True)
        return result

    def findings(self):
        lines = []
        p = self.previous.propagation
        for features, prefix, base in [('Local', 'Local', 'Local-only RF (93)'),
                                      ('All', 'Original', 'All-original RF (165)')]:
            baseline = p[p.Model == base]['PR-AUC'].mean()
            legacy = p[p.Model == prefix+' + legacy propagation RF']['PR-AUC'].mean()
            classical = p[p.Model.isin([prefix+' + legacy propagation RF', prefix+' + propagation RF'])]
            means = classical.groupby(['Model', 'Hops'])[['Validation PR-AUC', 'PR-AUC']].mean()
            winner = means.loc[means['Validation PR-AUC'].idxmax()]
            lines.append(f'**{features} → classical graph features:** fixed legacy RF AP {legacy:.3f} '
                         f'(mean gain {legacy-baseline:+.3f}); validation-selected graph RF AP '
                         f'{winner["PR-AUC"]:.3f} (gain {winner["PR-AUC"]-baseline:+.3f}). Gains depend on configuration.')
        old_pairs = pd.read_csv(self.directory / 'matched_graph_pairs.csv')
        numeric = old_pairs[old_pairs.Study == 'Propagation'].groupby('Comparison')['Δ PR-AUC'].mean()
        lines.append(f'**Quantum on numerical propagation:** mean paired AP differences span '
                     f'{numeric.min():+.3f} to {numeric.max():+.3f} across encoders/feature sets; no consistent boost.')
        probe = self.probes[self.probes.Case == 'Degree + triangle collision (8 nodes)']
        ham = probe[probe.Encoder == 'Quantum Hamiltonian'].iloc[0]
        spectral = probe[probe.Encoder == 'Classical spectral RBF'].iloc[0]
        invariant = probe[probe.Encoder == 'Quantum Hamiltonian invariant moments (exact)'].iloc[0]
        lines.append(f'**Direct topology probe:** Hamiltonian similarity is {ham["Different graph similarity"]:.3f} '
                     f'for the summary-collision pair; relabeled-same mean {ham["Isomorphic mean similarity"]:.3f}. '
                     f'Exact invariant moments give relabeled-same {invariant["Isomorphic mean similarity"]:.3f}, '
                     f'different-graph {invariant["Different graph similarity"]:.3f}. Classical spectra also distinguish it '
                     f'({spectral["Different graph similarity"]:.3f}); this is not a classical-impossibility result.')
        for features in ('Local', 'All'):
            paired = self.pairs[(self.pairs.Features == features) & (self.pairs.Axis == 'Architecture')]
            selected = paired[paired.Quantum == 'Raw + Hamiltonian quantum'].iloc[0]
            lines.append(f'**{features}, new matched pilot:** Hamiltonian minus adjacency-RBF AP '
                         f'{selected["Δ PR-AUC"]:+.3f}; inspect all five metrics below.')
        lines.append('**Interpretation:** topology sensitivity, permutation robustness, predictive gain and hardware feasibility '
                     'are separate tests. These small simulator pilots do not establish quantum advantage or classical intractability.')
        return lines

    def figure(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, features, prefix, base in [
            (axes[0, 0], 'Local', 'Local', 'Local-only RF (93)'),
            (axes[0, 1], 'All', 'Original', 'All-original RF (165)')]:
            frame = self.previous.propagation
            names = [base, prefix+' + propagation RF', prefix+' + propagation + RBF', prefix+' + propagation + hop quantum']
            values, errors = [], []
            for name in names:
                statistics = frame[frame.Model == name].groupby('Hops')['PR-AUC'].agg(['mean', 'std'])
                sd = statistics['std'].fillna(0)
                ax.errorbar(statistics.index, statistics['mean'], yerr=sd, marker='o', capsize=3,
                            label=ResultPlots.short_name(name))
                values.extend(statistics['mean']); errors.extend(sd)
            ax.set(title=f'{features}: adding graph information', xlabel='Hops', ylabel='Test AP (mean ± sampling SD)')
            ResultPlots.relevant_ylim(ax, values, errors)
            ax.legend(fontsize=7)
        probe = self.probes[self.probes.Case == 'Degree + triangle collision (8 nodes)']
        selected = probe[probe.Encoder.isin(['Classical summary RBF', 'Classical rooted WL',
            'Classical spectral RBF', 'Quantum Hamiltonian invariant moments (exact)',
            'Quantum Hamiltonian invariant moments (product)'])]
        ax = axes[1, 0]; x = np.arange(len(selected))
        ax.plot(x, selected['Isomorphic mean similarity'], 'o', label='Relabeled same graph')
        ax.plot(x, selected['Different graph similarity'], 's', label='Different graph')
        ax.set_xticks(x, selected.Encoder.replace({'Classical summary RBF':'Summary RBF', 'Classical rooted WL':'Rooted WL', 'Classical spectral RBF':'Spectral RBF', 'Quantum Hamiltonian invariant moments (exact)':'Q moments (exact)', 'Quantum Hamiltonian invariant moments (product)':'Q moments (product)'}), rotation=25, ha='right', fontsize=8)
        ax.set(title='Structural sensitivity is a separate test', ylabel='Kernel similarity')
        ResultPlots.relevant_ylim(ax, np.r_[selected['Isomorphic mean similarity'],selected['Different graph similarity']])
        ax.legend(fontsize=7)
        ax=axes[1, 1]
        paired = self.pairs[(self.pairs.Axis == 'Architecture') & self.pairs.Quantum.isin([
            'Raw + Graph message quantum', 'Raw + Hamiltonian quantum', 'Raw + Hermitian walk quantum'])]
        x=np.arange(len(paired)); values=paired['Δ PR-AUC']
        ax.axhline(0,color='gray',linestyle='--',linewidth=1)
        ax.plot(x,values,'o')
        ax.set_xticks(x, paired.Features+': '+paired.Quantum.str.replace('Raw + ', '', regex=False), rotation=25, ha='right', fontsize=8)
        ax.set(title='New pilot: quantum − same-input adjacency RBF', ylabel='Paired test AP difference')
        ResultPlots.relevant_ylim(ax,values,include_zero=True,bounded=False)
        for ax in axes.flat: ax.grid(alpha=.2)
        fig.tight_layout()
        return fig
