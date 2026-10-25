"""Classification and investigation-budget metrics shared by all experiments."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, roc_auc_score)


class Evaluation:
    CORE_METRICS = ('Accuracy', 'Balanced accuracy', 'Illicit F1', 'PR-AUC', 'ROC-AUC')

    @staticmethod
    def classification(y, predictions, scores):
        y = np.asarray(y)
        return {
            'Accuracy': accuracy_score(y, predictions),
            'Balanced accuracy': balanced_accuracy_score(y, predictions),
            'Illicit F1': f1_score(y, predictions, zero_division=0),
            'PR-AUC': average_precision_score(y, scores) if np.any(y == 1) else np.nan,
            'ROC-AUC': roc_auc_score(y, scores) if len(np.unique(y)) == 2 else np.nan,
        }

    @staticmethod
    def cumulative_discoveries(y, scores):
        return np.cumsum(np.asarray(y)[np.argsort(-np.asarray(scores), kind='stable')] == 1)

    @classmethod
    def inspections_to_find(cls, y, scores, n_illicit=10):
        if n_illicit < 1:
            raise ValueError('n_illicit must be positive')
        reached = np.flatnonzero(cls.cumulative_discoveries(y, scores) >= n_illicit)
        return int(reached[0] + 1) if len(reached) else np.nan

    @classmethod
    def discovery(cls, y, scores, budgets=(25, 50, 100, 200)):
        cumulative = cls.cumulative_discoveries(y, scores)
        rows = []
        for budget in budgets:
            if budget < 1:
                raise ValueError('Budgets must be positive')
            k = min(budget, len(cumulative))
            found = int(cumulative[k - 1]) if k else 0
            rows.append({'budget': k, 'found': found,
                         'precision': found / k if k else np.nan,
                         'recall': found / np.sum(np.asarray(y) == 1)
                         if np.any(np.asarray(y) == 1) else np.nan})
        return pd.DataFrame(rows)
