import random
import numpy as np
import pandas as pd


def calculate_metric(y, y_pred, metric):
    if metric == 'log_loss' or not metric:
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    elif metric == 'accuracy':
        y_pred_binary = (y_pred > 0.5)
        return np.mean(y == y_pred_binary)
    elif metric == 'precision':
        y_pred_binary = (y_pred > 0.5)
        true_positives = np.sum(y * y_pred_binary)
        predicted_positives = np.sum(y_pred_binary)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    elif metric == 'recall':
        y_pred_binary = (y_pred > 0.5)
        true_positives = np.sum(y * y_pred_binary)
        actual_positives = np.sum(y)
        return true_positives / actual_positives if actual_positives > 0 else 0.0
    elif metric == 'roc_auc':
        y_pred = np.round(y_pred, 10) # чтобы убрать влияние последовательности арифм операций на последовательность
        # аргументов, сильно влияющую на метрику
        # сортировка полученных значений вероятностей от большего к меньшему.
        trip = sorted(zip(y_pred, y), reverse=True)
        p = np.sum(y)  # число обьектов положительного класса
        n = len(y) - p  # число обьектов отрицательного класса
        auc = 0.0
        n_zeroes = n

        for _, true in trip:
            if true == 1:
                auc += n_zeroes
            else:
                n_zeroes -= 1
        auc /= p * n
        return auc
    elif metric == 'f1':
        precision = calculate_metric(y, y_pred, 'precision')
        recall = calculate_metric(y, y_pred, 'recall')
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def entropy(data):
    classes, class_count = np.unique(data, return_counts=True)  # количество уникальных классов
    S = 0  # Энтропия по Шеннону
    for count in class_count:
        p = count / np.sum(class_count)  # вероятность нахождения системы в этом классе
        if p != 0:  # чтобы логарифм работал
            S -= p * np.log2(p)
    return S


def gini_impurity(data):
    classes, class_count = np.unique(data, return_counts=True)  # количество уникальных классов
    return 1 - np.sum((class_count / np.sum(class_count)) ** 2)


# Для бинарных условий
def information_gain(data, mask, criterion):
    left, right = data[mask], data[~mask]
    if criterion == 'entropy':
        return entropy(data) - len(left) / len(data) * entropy(left) - len(right) / len(data) * entropy(right)
    elif criterion == 'gini':
        return gini_impurity(data) - len(left) / len(data) * gini_impurity(left) - len(right) / len(data) * gini_impurity(right)


class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.bins = bins
        self.criterion = criterion
        self.tree = None
        self.feature_bins = {}
        self.fi = {} # dict of feature importance

    def __str__(self):
        return f'MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}'

    def fit(self, X, y):
        self.leafs_cnt = 0
        if self.bins:
            self._build_histograms(X)
        for feature in X.columns:
            self.fi[feature] = 0

        self.tree = self._build_tree(X, y, depth=0)

        for feature in X.columns:
            self.fi[feature] /= len(X)

    def _build_tree(self, X, y, depth):
        if (len(np.unique(y)) == 1
                or len(y) == 1
                or depth == self.max_depth
                or len(y) < self.min_samples_split
                or (self.max_leafs - self.leafs_cnt == 1 and self.max_leafs != 1)
                or (self.max_leafs - self.leafs_cnt == 0 and self.max_leafs == 1)):
            self.leafs_cnt += 1
            return {'class': y.mean()}

        col_name, split_value, IG = self._get_best_split(X, y)

        mask = X[col_name] <= split_value
        self.fi[col_name] += len(X) * IG
        self.leafs_cnt += 1  # Для корректного построения с ограничением max_leafs нужно учитывать потенциальные листья
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
        if 'class' in node:
            return node['class']
        if row[node['col_name']] <= node['split_value']:
            return self._predict_proba_row(row, node['left'])
        else:
            return self._predict_proba_row(row, node['right'])

    def predict_proba(self, X):
        return [self._predict_proba_row(row) for _, row in X.iterrows()]

    def predict(self, X):
        probas = self.predict_proba(X)
        return [1 if p > 0.5 else 0 for p in probas]

    def print_tree(self, node=None, depth=0):
        if not node:
            node = self.tree
        if 'class' in node:
            print('  ' * depth + f'Class: {node["class"]}')
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
                cur_IG = information_gain(y, mask, self.criterion)
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
