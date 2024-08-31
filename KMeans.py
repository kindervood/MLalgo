import numpy as np
import pandas as pd

#  Итерационный алгоритм K-Means для задачи кластеризации
#  n_clusters – количество кластеров, которые нужно сформировать
#  max_iter – количество итераций алгоритма
#  n_init – сколько раз прогоняется алгоритм k-means
#  random_state – для воспроизводимости результата зафиксируем сид
class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_ = None  # координаты всех центроидов
        self.inertia_ = np.inf  # лучшее значение WCSS

    def __str__(self):
        s = 'MyKMeans class: '
        for attr, value in self.__dict__.items():
            s += f'{attr}={value}, '
        return s[:-2]

    @staticmethod
    def _euclidean_dist(dot_1, dot_2):
        return np.sqrt(np.sum((dot_1 - dot_2)**2))

    def fit(self, X):
        np.random.seed(seed=self.random_state)

        for _ in range(self.n_init):
            # Инициализация центроидов (можно использовать K-means++)
            cluster_centers = []
            for i in range(self.n_clusters):
                centroid = [np.random.uniform(np.min(X[feature]), np.max(X[feature])) for feature in X.columns]
                cluster_centers.append(centroid)

            for iteration in range(self.max_iter):
                clusters = [[] for _ in range(self.n_clusters)]

                # Определяем каждой точке(строке) в какой кластер она попадет
                for dot in X.values:
                    dist = [self._euclidean_dist(dot, centroid) for centroid in cluster_centers]  # дистанция до всех центров
                    cluster_ind = np.argmin(dist)
                    clusters[cluster_ind].append(dot)

                # Определение новых центроидов
                new_cluster_centers = [np.average(clusters[i], axis=0) if clusters[i] else cluster_centers[i] for i in range(self.n_clusters)]

                # Вычисление within-cluster sum of squares
                wcss = 0
                for ind, centroid in enumerate(new_cluster_centers):
                    for dot in clusters[ind]:
                        wcss += self._euclidean_dist(dot, centroid) ** 2

                # Проверка на сходимость
                if np.all(np.array(new_cluster_centers) == np.array(cluster_centers)):
                    break

                cluster_centers = new_cluster_centers

            if wcss < self.inertia_:
                self.inertia_ = wcss
                self.cluster_centers_ = cluster_centers

    def predict(self, X):
        cluster_indexes = []
        for dot in X.values:
            dist = [self._euclidean_dist(dot, centroid) for centroid in self.cluster_centers_]  # дистанция до всех центров
            cluster_ind = np.argmin(dist)
            cluster_indexes.append(cluster_ind)
        return cluster_indexes


# Пример использования

from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
X = pd.DataFrame(X)
X.columns = [f'col_{col}' for col in X.columns]

km = MyKMeans(10, 10, 3)
km.fit(X)
print(km.inertia_)
print(np.sum(km.cluster_centers_))
print(np.unique(km.predict(X), return_counts=True))
