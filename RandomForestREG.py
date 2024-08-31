import random
import numpy as np
import pandas as pd
from DecisonTreeREG import MyTreeReg


def calculate_metric(y, y_pred, metric):
    delta = y - y_pred
    if metric == 'mae':
        return np.mean(np.abs(delta))
    elif metric == 'mse':
        return np.mean(delta ** 2)
    elif metric == 'rmse':
        return np.sqrt(np.mean(delta ** 2))
    elif metric == 'mape':
        return 100 * np.mean(np.abs(delta / y))
    elif metric == 'r2':
        return 1 - np.mean(delta ** 2) / np.mean((y - np.average(y)) ** 2)

# Ансамблевый алгоритм случайного леса для задачи регрессии
# Также реализован подсчет важности фичей и Out of Bag ошибки, сохраненной в переменной класса oob_score_
# n_estimators – количество деревьев в лесу.
# max_features – доля фичей, которая будет случайным образом отбираться для каждого дерева. От 0.0 до 1.0
# max_samples – доля сэмплов, которая будет случайным образом отобрана из датасета для каждого дерева. От 0.0 до 1.0
# random_state – для воспроизводимости результата зафиксируем сид
# все параметры, присущие дереву решений: max_depth, min_samples_split, max_leafs, bins
class MyForestReg:
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.oob_score = oob_score  # type of oob score

        self.leafs_cnt = 0
        self.forest = []
        self.fi = {}  # feature importance
        self.oob_score_ = 0  # actual oob score

    def __str__(self):
        s = 'MyForestReg class: '
        for attr, value in self.__dict__.items():
            s += f'{attr}={value}, '
        return s[:-2]

    def fit(self, X, y):
        random.seed(self.random_state)
        for feature in X.columns:
            self.fi[feature] = 0

        oob_predictions = [None] * len(X)  # saving y_pred for calculating oob score

        for i in range(self.n_estimators):
            init_cols = list(X.columns.values)
            cols_smpl_cnt = round(len(X.columns) * self.max_features)
            init_rows_cnt = len(X)
            rows_smpl_cnt = round(len(X) * self.max_samples)
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)


            X_sample = X[cols_idx].iloc[rows_idx]
            y_sample = y.iloc[rows_idx]

            tree = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_leafs=self.max_leafs, bins=self.bins)
            tree.fit(X_sample, y_sample)

            for feature in X_sample.columns:
                self.fi[feature] += tree.fi[feature] / len(X) * len(X_sample) # нормируем по исходному датасету, а не по переданному

            self.forest.append(tree)
            self.leafs_cnt += tree.leafs_cnt

            # for oob score
            unused_rows_idx = list(set(range(init_rows_cnt)) - set(rows_idx))
            X_oob = X[cols_idx].iloc[unused_rows_idx]
            y_pred = tree.predict(X_oob)

            for idx, pred in zip(unused_rows_idx, y_pred):
                if oob_predictions[idx] is None:
                    oob_predictions[idx] = []
                oob_predictions[idx].append(pred)

        final_y_pred = []
        indexes = []
        for idx, preds in enumerate(oob_predictions):
            if preds:
                indexes.append(idx)
            if preds is not None:
                final_y_pred.append(np.average(preds))

        if self.oob_score:
            self.oob_score_ = calculate_metric(y[indexes], final_y_pred, self.oob_score)
        else:
            self.oob_score_ = None

    def predict(self, X):
        predictions = []
        for tree in self.forest:
            predictions.append(tree.predict(X))
        return np.average(predictions, axis=0)
