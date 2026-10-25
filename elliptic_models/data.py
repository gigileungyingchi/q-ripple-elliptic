"""Elliptic loading, chronological splits and reproducible subsampling."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


@dataclass
class EllipticData:
    transactions: pd.DataFrame
    edges: pd.DataFrame

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        features = pd.read_csv(directory / 'elliptic_txs_features.csv', header=None)
        if features.shape[1] != 167:
            raise ValueError('Expected txId, time_step and 165 feature columns')
        features.columns = ['txId', 'time_step'] + [f'x{i}' for i in range(165)]
        labels = pd.read_csv(directory / 'elliptic_txs_classes.csv')
        frame = features.merge(labels, on='txId', validate='one_to_one')
        label = frame['class'].astype(str).map({'1': 1, '2': 0, 'unknown': -1}).rename('label')
        frame = pd.concat([frame, label], axis=1)
        if frame.label.isna().any():
            raise ValueError('Unrecognized class label')
        return cls(frame, pd.read_csv(directory / 'elliptic_txs_edgelist.csv'))

    @staticmethod
    def feature_columns(name='Local', include_time=False):
        """Local excludes graph aggregates; All includes the 72 aggregate features."""
        if name not in ('Local', 'All'):
            raise ValueError('Feature set must be Local or All')
        return (['time_step'] if include_time else []) + [
            f'x{i}' for i in range(93 if name == 'Local' else 165)]

    def split(self, labelled_only=True):
        frame = self.transactions
        if labelled_only:
            frame = frame[frame.label >= 0]
        return {
            'train': frame[frame.time_step <= 30].copy(),
            'val': frame[frame.time_step.between(31, 34)].copy(),
            'test': frame[frame.time_step >= 35].copy(),
        }


class StratifiedSampler:
    @staticmethod
    def indices(y, size=None, seed=42):
        """Preserve natural class proportions rather than balancing the test set."""
        y = np.asarray(y)
        if size is None or size >= len(y):
            return np.arange(len(y))
        if size < 2 * len(np.unique(y)):
            raise ValueError('Sample too small for stratification')
        return np.sort(train_test_split(np.arange(len(y)), train_size=size,
                                        stratify=y, random_state=seed)[0])

    @staticmethod
    def balanced_indices(y, per_class, seed=42):
        rng = np.random.default_rng(seed)
        y = np.asarray(y)
        return np.concatenate([rng.choice(np.flatnonzero(y == c),
                               min(per_class, np.sum(y == c)), replace=False)
                               for c in np.unique(y)])

    @staticmethod
    def landmarks(y, count, seed=42):
        y = np.asarray(y)
        if not 1 <= count <= len(y):
            raise ValueError('Landmark count must be between 1 and training size')
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        chosen = []
        for c in classes:
            pool = np.flatnonzero(y == c)
            chosen.extend(rng.choice(pool, min(len(pool), count // len(classes)),
                                     replace=False).tolist())
        remaining = np.setdiff1d(np.arange(len(y)), chosen)
        chosen.extend(rng.choice(remaining, count - len(chosen), replace=False).tolist())
        return np.asarray(chosen, dtype=int)


class AnglePreprocessor:
    """Fit exclusively on training observations; PCA or full-feature angular encoding."""
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        if self.n_components is None:
            self.pipeline = StandardScaler().fit(X)
        else:
            self.pipeline = Pipeline([
                ('standard', StandardScaler()),
                ('pca', PCA(n_components=self.n_components, random_state=42)),
                ('angles', MinMaxScaler(feature_range=(0, np.pi), clip=True)),
            ]).fit(X)
        return self

    def transform(self, X):
        values = self.pipeline.transform(X)
        return 2 * np.arctan(values) if self.n_components is None else values

    def fit_transform(self, X):
        return self.fit(X).transform(X)
