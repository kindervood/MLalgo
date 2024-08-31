import random
import numpy as np
import pandas as pd
from DecisionTreeREG import MyTreeReg, calculate_metric

# Ансамблевый алгоритм стохастического градиентного бустинга над решающими деревьями для задачи регрессии
# Возможна поддержка изменяющейся learning_rate(lambda функция). Подсчитывается важность фичей
# Реализованы механизм early_stopping и регуляризация(штраф за количество листьев) для защиты от переобучения, 
# n_estimators – количество деревьев в лесу, learning_rate – скорость обучения, reg - отвечающий за регуляризацию
# все параметры, присущие дереву решений: max_depth, min_samples_split, max_leafs, bins,
# параметры для создания подвыборок для обучения: max_features, max_samples, random_state

class MyBoostReg:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5, min_samples_split=2, max_leafs=20, bins=16,
                 loss='mse', metric=None, max_features=0.5, max_samples=0.5, random_state=42, reg=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.loss = loss.lower()
        self.metric = metric.lower() if metric else None
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg

        self.pred_0 = None  # изначальное предсказание.
        self.trees = []  # список обученных деревьев
        self.best_score = None  # переменная для хранения лучшего/последнего скора
        self.total_leaf_count = 0  # количество листьев в обученных деревьях(для решуляризации)
        self.fi = {}  # важность фичей

    def __str__(self):
        s = 'MyBoostReg class: '
        for attr, value in self.__dict__.items():
            s += f'{attr}={value}, '
        return s[:-2]

    @staticmethod
    def _min_error_func(metric):
        if metric == 'mse':
            return np.mean
        elif metric == 'mae':
            return np.median

    # обучение на основе отклонений с механизмом early_stopping
    def fit(self, X, y, X_eval=None, y_eval=None, early_stopping=None, verbose=None):
        random.seed(self.random_state)

        for feature in X.columns:
            self.fi[feature] = 0

        self.pred_0 = self._min_error_func(self.loss)(y)
        cur_pred = np.full(y.shape, self.pred_0)

        best_eval_score = None  # Лучший скор на eval выборке
        no_improvement_count = 0  # количество раз неулучшения скора на eval выборке

        for i in range(1, self.n_estimators + 1):
            # если lr - функция, то используем i в качестве параметра. Ex: lambda i: 0.5 * (0.85 ** i)
            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate

            gradient = self._compute_gradient(y, cur_pred)

            tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)

            # формирование подвыборки для обучения
            init_cols = list(X.columns.values)
            cols_smpl_cnt = round(len(X.columns) * self.max_features)
            init_rows_cnt = len(X)
            rows_smpl_cnt = round(len(X) * self.max_samples)
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)

            X_sample = X[cols_idx].iloc[rows_idx]
            y_sample = -gradient[rows_idx]  # обучение на антиградиенте

            tree.fit(X_sample, y_sample)
            self._update_tree_predictions(tree, y, cur_pred)
            self.trees.append(tree)

            if X_eval is not None and y_eval is not None and early_stopping is not None:
                y_eval_pred = self.predict(X_eval)
                score = calculate_metric(y_eval, y_eval_pred, self.metric if self.metric else self.loss)

                if best_eval_score is None or score < best_eval_score:
                    best_eval_score = score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= early_stopping:  # скор не улучшается early_stopping раз
                    self.best_score = best_eval_score
                    self.trees = self.trees[:-early_stopping]  # удаляем деревья, которые не дали улучшения
                    break

            for feature in X_sample.columns:
                self.fi[feature] += tree.fi[feature] / len(X) * len(
                    X_sample)  # нормируем по исходному датасету, а не по переданному

            cur_pred += lr * np.array(tree.predict(X))

            if verbose is not None and i % verbose == 0:
                y_pred = self.predict(X)
                loss_value = calculate_metric(y, y_pred, self.loss)
                print(f'{i}. Loss[{self.loss}]: {loss_value}. best_eval_score:{best_eval_score}')

    def predict(self, X):
        if callable(self.learning_rate):
            predictions = np.full(X.shape[0], self.pred_0)
            for i, tree in enumerate(self.trees):
                predictions += self.learning_rate(i + 1) * np.array(tree.predict(X))
            return predictions
        return self.pred_0 + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)

    def _compute_gradient(self, y, y_pred):
        if self.loss == 'mse':
            return 2 * (y - y_pred)
        elif self.loss == 'mae':
            return np.sign(y - y_pred)

    def _update_tree_predictions(self, tree, y, cur_pred):
        # Обходим все листья дерева
        leaf_nodes = self._get_leaf_nodes(tree.tree)
        for leaf in leaf_nodes:
            # Находим все обьекты попавшие в лист, считаем для них разницу между исходным таргетом и текущим предсказанием
            diff = y[leaf['indexes']] - cur_pred[leaf['indexes']]
            # Подменяем антиградиент на значение, которое лучше всего уменьшает функцию потерь.
            # Применяем регуляризацию - штрафуем модель за количество листьев
            leaf['value'] = self._min_error_func(self.loss)(diff) + self.reg * self.total_leaf_count
        self.total_leaf_count += len(leaf_nodes)

    def _get_leaf_nodes(self, node):
        if 'value' in node:
            return [node]
        else:
            leaves = []
            if 'left' in node:
                leaves.extend(self._get_leaf_nodes(node['left']))
            if 'right' in node:
                leaves.extend(self._get_leaf_nodes(node['right']))
            return leaves
