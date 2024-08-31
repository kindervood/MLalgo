import numpy as np
import random
import pandas as pd
from sklearn.datasets import make_regression


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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
        sorted_indices = np.argsort(y_pred)[::-1]
        y_sorted = y[sorted_indices]
        y_pred = y_pred[sorted_indices]

        p = np.sum(y)  # число обьектов положительного класса
        n = len(y) - p  # число обьектов отрицательного класса
        auc = 0.0

        for i in range(len(y_sorted)):
            if y_sorted[i] == 0:  # Для каждого отрицательного
                # сколько положительных классов находятся выше текущего по скору
                above_positives = np.sum(y_sorted[:i] == 1)
                # сколько положительных классов имеют такой же скор
                same_score = np.sum(y_sorted[np.where(y_pred[:i] == y_pred[i])] == 1)
                auc += above_positives * same_score / 2

        auc /= p * n
        return auc
    elif metric == 'f1':
        precision = calculate_metric(y, y_pred, 'precision')
        recall = calculate_metric(y, y_pred, 'recall')
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


# "Классическая" логистическая регрессия с алгоритмом стохастического градиентного спуска, изменяющейся скоростью обучения learning_rate и регуляризацией elastic net
# Реализованы метрики: accuracy, precision, recall, f1, roc_auc(приближенно)
class MyLogReg:
    def __init__(self, n_iter, learning_rate=0.1, sgd_sample=None, reg=None, l1_coef=0, l2_coef=0, metric=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.sgd_sample = sgd_sample if sgd_sample else 1.0
        self.metric = metric
        self.weights = []
        self.best_score = 0
        self.reg = reg
        self.l1_coef = l1_coef if reg == 'l1' or reg == 'elasticnet' else 0
        self.l2_coef = l2_coef if reg == 'l2' or reg == 'elasticnet' else 0
        self.random_state = random_state

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X_data, y, verbose=None):
        X = X_data.copy()
        eps = 1e-15
        random.seed(self.random_state)
        X.insert(0, 'col_0', 1)

        if type(self.sgd_sample) == float:
            self.sgd_sample = int(self.sgd_sample * X.shape[0])

        n_params = len(X.columns)
        self.weights = np.array([1.0] * n_params)

        for i in range(1, self.n_iter + 1):
            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate

            sample_rows_idx = list(random.sample(range(X.shape[0]), self.sgd_sample))
            X_sample = X.iloc[sample_rows_idx]
            y_sample = y.iloc[sample_rows_idx]

            y_pred = sigmoid(np.dot(X_sample, self.weights))
            delta = y_pred - y_sample

            log_loss = -np.mean(y_sample * np.log(y_pred + eps) + (1 - y_sample) * np.log(1 - y_pred + eps))
            loss = log_loss + self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)

            gradient_loss = np.dot(X_sample.T, delta) / X_sample.shape[0] + self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

            self.weights -= gradient_loss * lr

            if verbose and i % verbose == 0:
                if self.metric:
                    metric_score = calculate_metric(y_sample, y_pred, self.metric) if self.metric else loss
                    print(f"{i}|loss:{loss}|{self.metric}:{metric_score}")
                else:
                    print(f"{i}|loss:{loss}")

        final_y_pred = sigmoid(np.dot(X, self.weights))
        self.best_score = calculate_metric(y, final_y_pred, self.metric)

    def get_coef(self):
        return np.array(self.weights[1:])

    def predict(self, X_data):
        X = X_data.copy()
        X.insert(0, 'col_0', 1)
        y_pred = sigmoid(np.dot(X, self.weights))
        return (y_pred > 0.5).astype(int)

    def predict_proba(self, X_data):
        X = X_data.copy()
        X.insert(0, 'col_0', 1)
        y_pred = sigmoid(np.dot(X, self.weights))
        return y_pred

    def get_best_score(self):
        return self.best_score
