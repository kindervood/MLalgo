import numpy as np
import pandas as pd
from scipy.stats import mode
from scipy.spatial.distance import cosine

# Взвешенный KNN для классификации
# Возможные веса uniform(у всех вес=1), rank(на основе порядкового номера), distance(на основе дистанции)
# Реализованы метрики: euclidean, chebyshev, manhattan, cosine
class MyKNNClf:
    def __init__(self, k=1, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.X_train = None
        self.y_train = None
        self.metric = metric
        self.weight = weight

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape

    # нахождение k ближайших по метрике соседей
    def k_nearest_neighbours(self, dot):
        dist = self.distances(dot)
        nearest_indices = np.argsort(dist)[:self.k]
        if self.weight == 'uniform':
            weights = np.array([1] * self.k)
        elif self.weight == 'rank':
            weights = 1 / (np.arange(1, self.k + 1))
        elif self.weight == 'distance':
            weights = 1 / dist[nearest_indices]  # Add small epsilon to avoid division by zero?
        return self.y_train.iloc[nearest_indices], weights

    # Дистанции до точки от тренировочного набора для заданной метрики
    def distances(self, dot):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - dot) ** 2, axis=1))
        elif self.metric == 'chebyshev':
            return np.max(np.abs(self.X_train - dot), axis=1)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - dot), axis=1)
        elif self.metric == 'cosine':
            return np.array([cosine(x, dot) for x in self.X_train.values])

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            nearest_labels, weights = self.k_nearest_neighbours(X.iloc[i])
            class_0_weight = np.sum(weights * (nearest_labels == 0)) / np.sum(weights)
            class_1_weight = np.sum(weights * (nearest_labels == 1)) / np.sum(weights)
            predictions.append(class_0_weight <= class_1_weight)
        return np.array(predictions)

    def predict_proba(self, X):
        probabilities = []
        for i in range(X.shape[0]):
            nearest_labels, weights = self.k_nearest_neighbours(X.iloc[i])
            class_1_weight = np.sum(weights * (nearest_labels == 1)) / np.sum(weights)
            probabilities.append(class_1_weight)
        return np.array(probabilities)
