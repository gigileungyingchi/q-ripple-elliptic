"""Consistent result tables and reusable, exportable experiment figures."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .metrics import Evaluation


class ResultReport:
    metrics = list(Evaluation.CORE_METRICS)

    @classmethod
    def table(cls, results, identifiers=(), extras=('Validation PR-AUC', 'Seconds')):
        """Require all five core metrics; diagnostics are optional extra columns."""
        missing = set(cls.metrics) - set(results.columns)
        if missing:
            raise ValueError(f'Missing core metrics: {sorted(missing)}')
        columns = list(dict.fromkeys([*identifiers, *cls.metrics,
                                     *(c for c in extras if c in results)]))
        return results[columns].round(4)

    @classmethod
    def summary(cls, results, by, extras=('Validation PR-AUC', 'Seconds'), uncertainty=True):
        """Compact mean ± sample SD; singleton groups show their value only."""
        cls.table(results)
        columns = cls.metrics + [c for c in extras if c in results and c not in cls.metrics]
        grouped = results.groupby(list(by), dropna=False, sort=False)[columns]
        means = grouped.mean()
        if not uncertainty:
            return means.round(4)
        deviations, counts = grouped.std(), grouped.count()
        formatted = means.copy().astype(object)
        for column in columns:
            formatted[column] = [
                '—' if pd.isna(value) else
                f'{value:.3f} ± {sd:.3f}' if count > 1 else f'{value:.3f}'
                for value, sd, count in zip(means[column], deviations[column], counts[column])]
        return formatted


class ResultPlots:
    metrics = list(Evaluation.CORE_METRICS)

    @staticmethod
    def short_name(name):
        replacements = {
            'Local-only RF (93)': 'Local 93 RF',
            'All-original RF (165)': 'All 165 RF',
            'Original + legacy propagation RF': 'All + legacy propagation',
            'Original + propagation RF': 'All + propagation',
            'Original + propagation + RBF': 'All + propagation + RBF',
            'Original + propagation + hop quantum': 'All + propagation + hop Q',
            'Original + propagation + generic quantum': 'All + propagation + generic Q',
            'Original + graph + RBF': 'All + graph + RBF',
            'Original + graph + WL': 'All + graph + WL',
            'Original + graph + quantum (directed)': 'All + graph + directed Q',
            'Original + graph + quantum (reverse)': 'All + graph + reverse Q',
            'Original + graph + quantum (undirected)': 'All + graph + undirected Q',
            'Original + graph + quantum (none)': 'All + graph + edgeless Q',
            'Original + graph: original ordering': 'Full inputs, original order',
            'Original + graph: permutation mixture': 'Full inputs, perm. mixture',
            'Original + graph, permuted test slots': 'Full inputs, permuted test',
            'Random Forest (raw)': 'RF raw', 'Random Forest (PCA/angles)': 'RF PCA',
            'Random Forest (transaction only)': 'RF transactions',
            'Quantum ZZ (original baseline)': 'Quantum ZZ',
            'Propagation Random Forest': 'Propagation RF',
            'Propagation RBF Nyström': 'Propagation RBF',
            'Hop-aware quantum Nyström': 'Hop-aware quantum',
            'Generic quantum Nyström': 'Generic quantum',
            'Retained-subgraph Random Forest': 'Subgraph RF',
            'Retained-subgraph RBF Nyström': 'Subgraph RBF',
            'WL on retained subgraphs': 'Subgraph WL',
            'Topology quantum: original ordering': 'Original ordering',
            'Topology quantum: permutation mixture': 'Permutation mixture',
            'Original model, permuted test slots': 'Permuted test slots',
        }
        name = str(name)
        if name.startswith('Quantum passes='):
            p = name.split('passes=')[1].split(',')[0]
            return f'Quantum p{p} ' + ('Nyström' if name.endswith('True') else 'exact')
        return replacements.get(name, name.replace('Topology quantum ', 'Quantum ').replace('Local + propagation + hop quantum', 'Local + propagation + hop Q').replace('Local + propagation + generic quantum', 'Local + propagation + generic Q'))

    @staticmethod
    def relevant_ylim(ax, values, errors=None, include_zero=False, bounded=True):
        """Fit the visible range to points AND error bars, with modest padding.

        Keep full uncertainty even if mean ± SD extends outside [0, 1]. For
        deltas, include zero explicitly and disable score-domain clipping.
        """
        values = np.asarray(values, dtype=float).ravel()
        errors = np.zeros_like(values) if errors is None else np.asarray(errors, dtype=float).ravel()
        valid = np.isfinite(values) & np.isfinite(errors)
        if not valid.any():
            return
        low = float(np.min(values[valid] - np.abs(errors[valid])))
        high = float(np.max(values[valid] + np.abs(errors[valid])))
        if include_zero:
            low, high = min(low, 0.), max(high, 0.)
        padding = max(.10 * (high - low), .005)
        lower, upper = low - padding, high + padding
        if bounded and low >= 0 and high <= 1:
            lower, upper = max(0., lower), min(1., upper)
        ax.set_ylim(lower, upper)

    @classmethod
    def metric_grid(cls, results, x='Hops', series='Model', metrics=None, title=''):
        """One panel per metric, with SD bars when repeated runs are available."""
        metrics = cls.metrics if metrics is None else list(metrics)
        columns = min(3, len(metrics))
        rows = int(np.ceil(len(metrics) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(4.7 * columns, 3.2 * rows), squeeze=False)
        for ax, metric in zip(axes.flat, metrics):
            values, errors = [], []
            for label, group in results.groupby(series, sort=False, dropna=False):
                statistics = group.groupby(x)[metric].agg(['mean', 'std']).sort_index()
                values.extend(statistics['mean'].tolist())
                errors.extend(statistics['std'].fillna(0).tolist())
                ax.errorbar(statistics.index, statistics['mean'],
                            yerr=statistics['std'].fillna(0), marker='o', capsize=3,
                            linewidth=1.5, label=cls.short_name(label))
            ax.set(xlabel=x, ylabel=metric, title=metric)
            cls.relevant_ylim(ax, values, errors)
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(metrics):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)), fontsize=8)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, .10, 1, .94))
        return fig

    @classmethod
    def metric_heatmap(cls, results, by=('Model',), title='', sort_validation=False):
        """All scores share the 0–1 color scale; optional ordering uses validation."""
        columns = cls.metrics + (['Validation PR-AUC'] if 'Validation PR-AUC' in results else [])
        summary = results.groupby(list(by), sort=False, dropna=False)[columns].mean()
        if sort_validation and 'Validation PR-AUC' in summary:
            summary = summary.sort_values('Validation PR-AUC', ascending=False)
        values = summary[cls.metrics]
        labels = [' / '.join(cls.short_name(v) for v in idx) if isinstance(idx, tuple)
                  else cls.short_name(idx) for idx in values.index]
        fig, ax = plt.subplots(figsize=(max(11, 8 + .08 * max(map(len, labels), default=0)), max(2.8, .4 * len(values) + 1.3)))
        plot = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(cls.metrics)), cls.metrics, rotation=15, ha='right', fontsize=9)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        for i in range(len(values)):
            for j in range(len(cls.metrics)):
                value = values.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=8,
                            color='black' if value > .6 else 'white')
        ax.set_title(title)
        fig.colorbar(plot, ax=ax, label='Test score', shrink=.8)
        fig.tight_layout()
        return fig

    @classmethod
    def ablation_deltas(cls, results, features=('Local', 'All')):
        """Paired per-seed deltas for the four quantum ablation branches."""
        rows = results[results.Model.str.startswith('Quantum passes=')].copy()
        base = rows[(rows.Passes == 1) & (rows['Nyström'] == False)]
        keys = ['Features', 'Seed']
        base = base[keys + cls.metrics].rename(columns={m: f'Base {m}' for m in cls.metrics})
        rows = rows.merge(base, on=keys, validate='many_to_one')
        for metric in cls.metrics:
            rows[f'Δ {metric}'] = rows[metric] - rows[f'Base {metric}']
        maximum = max(float(rows[[f'Δ {m}' for m in cls.metrics]].abs().max().max()), .001)
        fig, axes = plt.subplots(len(features), len(cls.metrics),
                                 figsize=(16, 2.9 * len(features)), squeeze=False)
        for i, feature in enumerate(features):
            part = rows[rows.Features == feature]
            for j, metric in enumerate(cls.metrics):
                ax = axes[i, j]
                matrix = part.pivot_table(index='Passes', columns='Nyström', values=f'Δ {metric}')
                plot = ax.imshow(matrix, cmap='RdBu_r',
                                 norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum), aspect='auto')
                ax.set_xticks(range(len(matrix.columns)), ['On' if c else 'Off' for c in matrix.columns])
                ax.set_yticks(range(len(matrix.index)), [str(int(v)) for v in matrix.index])
                ax.set(xlabel='Nyström', ylabel=f'{feature}: passes', title=f'Δ {metric}')
                for r in range(len(matrix)):
                    for c in range(len(matrix.columns)):
                        ax.text(c, r, f'{matrix.iloc[r, c]:+.3f}', ha='center', va='center', fontsize=9)
        fig.suptitle('Paired test changes from one-pass exact quantum kernel', fontsize=12)
        fig.tight_layout(rect=(0, 0, .94, .93))
        bar = fig.add_axes((.95, .20, .012, .58))
        fig.colorbar(plot, cax=bar, label='Score change')
        return fig

    @classmethod
    def runtime_tradeoff(cls, results, title=''):
        """Measured local time versus AP and balanced accuracy; log time axis."""
        summary = results.groupby('Model', sort=False)[cls.metrics + ['Seconds']].mean()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, metric in zip(axes, ['PR-AUC', 'Balanced accuracy']):
            for i, (name, row) in enumerate(summary.iterrows()):
                quantum = str(name).startswith('Quantum')
                color = 'tab:purple' if quantum else 'tab:blue'
                ax.scatter(max(row.Seconds, 1e-6), row[metric], color=color, s=45,
                           marker='s' if quantum else 'o')
                ax.annotate(str(i + 1), (max(row.Seconds, 1e-6), row[metric]),
                            xytext=(4, 4 + 6 * (i % 2)), textcoords='offset points', fontsize=8)
            ax.set(xscale='log', xlabel='Measured seconds (log scale)', ylabel=metric)
            cls.relevant_ylim(ax, summary[metric])
            ax.margins(x=.35)
            ax.grid(alpha=.2)
        from matplotlib.lines import Line2D
        keys = [Line2D([], [], linestyle='',
                marker='s' if str(name).startswith('Quantum') else 'o',
                color='tab:purple' if str(name).startswith('Quantum') else 'tab:blue',
                label=f'{i + 1}. {cls.short_name(name)}')
                for i, name in enumerate(summary.index)]
        fig.legend(handles=keys, loc='lower center', ncol=3, fontsize=8)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, .17, 1, .94))
        return fig

    @staticmethod
    def retention(retention):
        """Reachable versus retained node distributions and truncation rate."""
        test = retention[retention.Split == 'test']
        depths = sorted(test.Hops.unique())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        arrays = [test.loc[test.Hops == h, 'Reachable nodes'].to_numpy() for h in depths]
        axes[0].boxplot(arrays, tick_labels=[str(h) for h in depths], showfliers=False)
        retained = test.groupby('Hops')['Retained nodes'].mean().reindex(depths)
        axes[0].plot(range(1, len(depths) + 1), retained, marker='s', label='Mean retained')
        axes[0].set(yscale='log', xlabel='Hops', ylabel='Nodes (log scale)',
                    title='Test neighborhoods; outliers hidden')
        axes[0].legend(fontsize=8)
        means = test.groupby('Hops')[['Retention fraction', 'Truncated']].mean()
        axes[1].plot(means.index, means['Retention fraction'], marker='o', label='Mean fraction retained')
        axes[1].plot(means.index, means.Truncated, marker='s', label='Fraction of graphs truncated')
        axes[1].set(xlabel='Hops', ylabel='Fraction', ylim=(0, 1), title='Effect of the node cap')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=.2)
        fig.tight_layout()
        return fig

    @staticmethod
    def save(fig, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix('.png'), dpi=160, bbox_inches='tight')
        fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')


class ExperimentSummary:
    """A compact overview from saved results; requires no model retraining."""
    def __init__(self, directory):
        from pathlib import Path
        directory = Path(directory)
        files = {
            'classification': 'classification.csv', 'propagation': 'directional_propagation.csv',
            'topology': 'direct_topology.csv', 'coverage': 'seed_exposure_coverage.csv',
            'retention': 'topology_retention.csv', 'permutations': 'permutation_kernel_diagnostic.csv',
            'reference': 'graph_models.csv',
        }
        for name, filename in files.items():
            path = directory / filename
            if not path.exists():
                raise FileNotFoundError(f'Run the experiment sections first: missing {path.name}')
            setattr(self, name, pd.read_csv(path))
        self.local = self.classification[self.classification.Features == 'Local']

    def winners(self):
        """Validation-selected rows; no ranking of unlike studies by test score."""
        rows = []
        for study, frame, groups in [
            ('Local classification', self.local, ['Model']),
            ('Original + propagation', self.propagation[self.propagation.Model != 'Local-only RF (93)'], ['Model', 'Hops']),
            ('Original + topology', self.topology[self.topology.Model != 'Local-only RF (93)'], ['Model', 'Hops']),
        ]:
            summary = frame.groupby(groups)[ResultReport.metrics + ['Validation PR-AUC']].mean().reset_index()
            winner = summary.loc[summary['Validation PR-AUC'].idxmax()]
            rows.append({'Study': study, **winner.to_dict()})
        return pd.DataFrame(rows)

    def comparison_winners(self):
        """Best classical and quantum configuration per study and feature set.

        Rank by mean validation AP across seeds, never by test performance.
        Separate family winners may have different depths; matched comparisons
        below the summary table hold depth fixed when assessing kernel gains.
        """
        rows = []
        for study, frame, feature_column in [
            ('Classification', self.classification, 'Features'),
            ('Propagation', self.propagation, 'Original features'),
            ('Topology', self.topology, 'Original features'),
        ]:
            for features in ('Local', 'All'):
                subset = frame[frame[feature_column] == features].copy()
                # The All-topology study also lists a Local-only reference.
                if features == 'All' and study != 'Classification':
                    subset = subset[~subset.Model.str.startswith('Local')]
                quantum = subset.Model.str.contains('quantum', case=False, regex=False)
                for family, mask in [('Classical', ~quantum), ('Quantum', quantum)]:
                    candidates = subset[mask]
                    if candidates.empty:
                        continue
                    groups = ['Model'] + (['Hops'] if 'Hops' in candidates else [])
                    means = candidates.groupby(groups)[
                        ResultReport.metrics + ['Validation PR-AUC']].mean().reset_index()
                    winner = means.loc[means['Validation PR-AUC'].idxmax()]
                    rows.append({'Study': study, 'Features': features + (' 93' if features == 'Local' else ' 165'),
                                 'Family': family, **winner.to_dict()})
        return pd.DataFrame(rows)

    def findings(self):
        reference = self.reference[~self.reference.Diagnostic].set_index('Model')
        local = reference.loc['Local-only RF (93)', 'PR-AUC']
        original = reference.loc['All-original RF (165)', 'PR-AUC']
        legacy = reference.loc['Original + legacy propagation RF', 'PR-AUC']
        winners = self.winners().set_index('Study')
        propagation = winners.loc['Original + propagation']
        topology = winners.loc['Original + topology']
        baseline_name = 'Local-only RF (93)' if propagation.Model.startswith('Local') else 'All-original RF (165)'
        baseline = self.propagation[self.propagation.Model == baseline_name]['PR-AUC'].mean()
        last = self.coverage[self.coverage.Split == 'test'].sort_values('Hops').iloc[-1]
        return [
            f'**Restored matched reference:** Local 93 AP {local:.3f}; All 165 AP {original:.3f}; '
            f'All 165 + original propagation AP {legacy:.3f}.',
            f'**Additive propagation:** validation selects {ResultPlots.short_name(propagation.Model)}, '
            f'{int(propagation.Hops)} added hops; test AP {propagation["PR-AUC"]:.3f} '
            f'(mean delta versus its raw-feature baseline {propagation["PR-AUC"] - baseline:+.3f}).',
            f'**Additive topology:** validation selects {ResultPlots.short_name(topology.Model)}, '
            f'{int(topology.Hops)} hops; test AP {topology["PR-AUC"]:.3f}. '
            'Hybrid scores include the unchanged classical inputs; inspect paired gains to assess quantum value.',
            f'**Remaining limits:** seed exposure reaches {last["Any exposure through depth"]:.1%} '
            f'of test targets by {int(last.Hops)} hops. The quantum topology cap still approximates '
            'the graph; original attributes and full-period descriptors bypass it.',
        ]

    def figure(self):
        """Six panels: classification, ablation, depth, topology, coverage, cap."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        ax = axes[0, 0]
        reference = self.reference[~self.reference.Diagnostic].set_index('Model')
        names = ['Local-only RF (93)', 'Local + legacy propagation RF', 'All-original RF (165)', 'Original + legacy propagation RF']
        values = reference.loc[names, 'PR-AUC']
        ax.plot(range(4), values, 'o')
        ax.set_xticks(range(4), ['Local 93', 'Local + legacy', 'All 165', 'All + legacy'], rotation=25)
        ax.set(title='Restored reference: identical targets', ylabel='Test AP')
        ResultPlots.relevant_ylim(ax, values)

        ax = axes[0, 1]
        baseline = self.propagation[self.propagation.Model == 'All-original RF (165)'][['Seed', 'Hops', 'PR-AUC']]
        changes = self.propagation.merge(baseline.rename(columns={'PR-AUC': 'Base AP'}), on=['Seed', 'Hops'])
        changes['Delta AP'] = changes['PR-AUC'] - changes['Base AP']
        means, deviations = [], []
        names = ['Original + legacy propagation RF', 'Original + propagation RF',
                 'Original + propagation + RBF', 'Original + propagation + hop quantum']
        # One depth per model, selected using mean validation AP across seeds.
        for name in names:
            candidate = changes[changes.Model == name]
            chosen = candidate.groupby('Hops')['Validation PR-AUC'].mean().idxmax()
            deltas = candidate[candidate.Hops == chosen]['Delta AP']
            means.append(deltas.mean()); deviations.append(deltas.std() if len(deltas) > 1 else 0.)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.errorbar(range(4), means, yerr=deviations, fmt='o', capsize=4)
        ax.set_xticks(range(4), ['Legacy', 'Propagation', '+ RBF', '+ Hop Q'], rotation=15)
        ax.set(title='Added value versus All 165 (val-selected)', ylabel='Paired change in test AP')
        ResultPlots.relevant_ylim(ax, means, deviations, include_zero=True, bounded=False)

        for ax, frame, names, title, repeated in [
            (axes[0, 2], self.propagation,
             ['Local-only RF (93)', 'Local + propagation RF', 'All-original RF (165)', 'Original + propagation RF'],
             'Original inputs + added propagation', True),
            (axes[1, 0], self.topology,
             ['Local + graph + RBF', 'Local + graph + quantum (directed)',
              'Original + graph + RBF', 'Original + graph + quantum (directed)'],
             'Original inputs + graph additions', False),
        ]:
            values, errors = [], []
            for name in names:
                statistics = frame[frame.Model == name].groupby('Hops')['PR-AUC'].agg(['mean', 'std'])
                sd = statistics['std'].fillna(0) if repeated else np.zeros(len(statistics))
                ax.errorbar(statistics.index, statistics['mean'], yerr=sd,
                            marker='o', capsize=3, label=ResultPlots.short_name(name))
                values.extend(statistics['mean'])
                errors.extend(sd)
            ax.set(title=title, xlabel='Hops', ylabel='Test AP')
            ResultPlots.relevant_ylim(ax, values, errors)
            ax.legend(fontsize=7, loc='best')

        ax = axes[1, 1]
        coverage = self.coverage[self.coverage.Split == 'test'].sort_values('Hops')
        values = 100 * coverage['Any exposure through depth']
        ax.plot(coverage.Hops, values, marker='o', color='tab:green')
        ax.set(title='Seed coverage: all test targets', xlabel='Maximum walk length', ylabel='Targets with seed exposure (%)')
        ResultPlots.relevant_ylim(ax, values, bounded=False)

        ax = axes[1, 2]
        retention = self.retention[self.retention.Split == 'test'].groupby('Hops')['Retention fraction'].mean()
        ax.plot(retention.index, 100 * retention, marker='o', color='tab:orange')
        ax.set(title='Quantum branch: retained topology', xlabel='Neighborhood radius', ylabel='Mean reachable nodes retained (%)')
        ResultPlots.relevant_ylim(ax, 100 * retention, bounded=False)
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.suptitle('Results at a glance · simulator pilots · error bars = sampling SD', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .95))
        return fig
