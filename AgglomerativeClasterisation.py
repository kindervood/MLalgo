import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

# Агломеративный(снизу вверх) алгоритм иерархической кластеризации
# Реализованы метрики: euclidean, chebyshev, manhattan, cosine
# n_clusters – количество кластеров, которые нужно сформировать.
# metric - метрика для подсчета расстояния
class MyAgglomerative:
    def __init__(self, n_clusters=3, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric

    def __str__(self):
        s = 'MyAgglomerative class: '
        for attr, value in self.__dict__.items():
            s += f'{attr}={value}, '
        return s[:-2]

    @staticmethod
    def _find_dist(dot_1, dot_2, metric):
        if metric == 'euclidean':
            return np.sqrt(np.sum((dot_1 - dot_2) ** 2))
        elif metric == 'chebyshev':
            return np.max(np.abs(dot_1 - dot_2))
        elif metric == 'manhattan':
            return np.sum(np.abs(dot_1 - dot_2))
        elif metric == 'cosine':
            return cosine(dot_1 - dot_2)

    def fit_predict(self, X):
        n = X.shape[0]
        distance_matrix = np.zeros((n, n))
        clusters = [{'points': [i], 'centroid': X.values[i]} for i in range(n)]
        while len(clusters) > self.n_clusters:

            # Вычисление матрицы расстояний
            for i in range(len(clusters)):
                for j in range(i, len(clusters)):
                    dist = self._find_dist(clusters[i]['centroid'], clusters[j]['centroid'], self.metric)
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = np.inf

            # индексы двух ближайших точек
            min_dist = np.inf
            min_i, min_j = -1, -1
            for i in range(len(clusters)):
                for j in range(i, len(clusters)):
                    if distance_matrix[i, j] < min_dist:
                        min_dist = distance_matrix[i, j]
                        min_i, min_j = i, j

            # обьединяем ближайщие точки в один кластер
            new_points = clusters[min_i]['points'] + clusters[min_j]['points']
            new_centroid = np.mean(X.values[new_points], axis=0)
            new_cluster = {'points': new_points, 'centroid': new_centroid}

            clusters.pop(min_j)
            clusters.pop(min_i)
            clusters.append(new_cluster)

        labels = np.zeros(n)
        for i, cluster in enumerate(clusters):
            for point in cluster['points']:
                labels[point] = i

        return labels

# пример использрования
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
X = pd.DataFrame(X)
X.columns = [f'col_{col}' for col in X.columns]

cl = MyAgglomerative(n_clusters=10)
print(np.unique(cl.fit_predict(X), return_counts=True))
