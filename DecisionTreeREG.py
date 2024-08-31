import random
import numpy as np
import pandas as pd


def mse(data):
    n = len(data)
    if n != 0:
        return 1 / n * np.sum((data - np.average(data)) ** 2)
    return 0


# Для бинарных условий
def information_gain(data, mask):
    left, right = data[mask], data[~mask]
    return mse(data) - len(left) / len(data) * mse(left) - len(right) / len(data) * mse(right)

# Дерево решений для задачи регрессии
# Аналогично задаче классификации, разница лишь в поиске лучшего сплита(на основе mse), подсчете важности фичей, подсчете предсказания
class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.bins = bins
        self.tree = None
        self.feature_bins = {}
        self.fi = {} # dict of feature importance

    def __str__(self):
        return f'MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}'

    def fit(self, X, y):
        self.leafs_cnt = 0
        if self.bins:
            self._build_histograms(X)
        for feature in X.columns:
            self.fi[feature] = 0

        self.tree = self._build_tree(X, y, depth=0)

        for feature in X.columns:
            self.fi[feature] /= len(X)
            if self.fi[feature] == 0.0:
                self.fi[feature] = 0

    def _build_tree(self, X, y, depth):
        if (len(np.unique(y)) == 1
        or len(y) == 1
        or depth == self.max_depth
        or len(y) < self.min_samples_split
        or (self.max_leafs - self.leafs_cnt == 1 and self.max_leafs != 1)
        or (self.max_leafs - self.leafs_cnt == 0 and self.max_leafs == 1)):
            self.leafs_cnt += 1
            return {'value': y.mean()}

        col_name, split_value, IG = self._get_best_split(X, y)

        mask = X[col_name] <= split_value
        self.fi[col_name] += len(X) * IG
        self.leafs_cnt += 1 # Для корректного построения с ограничением max_leafs нужно учитывать потенциальные листья
        left_branch = self._build_tree(X[mask], y[mask], depth + 1)
        self.leafs_cnt -= 1
        right_branch = self._build_tree(X[~mask], y[~mask], depth + 1)

        return {
            'col_name': col_name,
            'split_value': split_value,
            'left': left_branch,
            'right': right_branch
        }

    def _predict_proba_row(self, row, node=None):
        if node is None:
            node = self.tree
        if 'value' in node:
            return node['value']
        if row[node['col_name']] <= node['split_value']:
            return self._predict_proba_row(row, node['left'])
        else:
            return self._predict_proba_row(row, node['right'])

    def predict(self, X):
        return [self._predict_proba_row(row) for _, row in X.iterrows()]

    def print_tree(self, node=None, depth=0):
        if not node:
            node = self.tree
        if 'value' in node:
            print('  ' * depth + f'value: {node["value"]}')
        else:
            print('  ' * depth + f'Split on {node["col_name"]} > {node["split_value"]}')
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)

    def _get_best_split(self, X, y):
        col_name, split_value, max_IG = '', 0, -np.inf
        for feature in X.columns:
            if self.bins:
                thresholds = self.feature_bins[feature]
            else:
                thresholds = np.unique(X[feature])

            for threshold in thresholds:
                mask = X[feature] <= threshold
                cur_IG = information_gain(y, mask)
                if cur_IG > max_IG:
                    col_name, split_value, max_IG = feature, threshold, cur_IG
        return col_name, split_value, max_IG

    def _build_histograms(self, X):
        for feature in X.columns:
            unique_values = np.unique(X[feature])
            if len(unique_values) <= self.bins - 1:
                self.feature_bins[feature] = unique_values
            else:
                counts, bin_edges = np.histogram(X[feature], bins=self.bins)
                self.feature_bins[feature] = bin_edges[1:-1]  # Убираем внешние границы
