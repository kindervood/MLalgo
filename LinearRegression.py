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


# "Классическая" линейная регрессия с алгоритмом стохастического градиентного спуска, изменяющейся скоростью обучения learning_rate и регуляризацией elastic net
class MyLineReg:
    def __init__(self, n_iter, learning_rate=0.1, sgd_sample=None, reg=None, l1_coef=0, l2_coef=0, metric=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.sgd_sample = sgd_sample if sgd_sample else 1.0 # может быть как целым числом, так и дробным от 0 до 1
        self.metric = metric
        self.weights = []
        self.best_score = 0
        self.reg = reg
        self.l1_coef = l1_coef if reg == 'l1' or reg == 'elasticnet' else 0
        self.l2_coef = l2_coef if reg == 'l2' or reg == 'elasticnet' else 0
        self.random_state = random_state # для воспроизводимости обучения

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    # X - фичи, y - целевая переменная, verbose - число итерация для вывода лога
    def fit(self, X, y, verbose=None):
        random.seed(self.random_state)
        # добавляем еденичный столбец к X слева
        X.insert(0, 'col_0', 1)

        # если задано в виде доли, то перевод в целое число(округленное)
        if type(self.sgd_sample) == float:
            self.sgd_sample = int(self.sgd_sample * X.shape[0])

        n_params = len(X.columns)
        self.weights = np.array([1.0] * n_params)

        for i in range(1, self.n_iter + 1):
            # если self.learning_rate функция, используй ее как функцию(для уменьшения lr при увеличении числа итераций)
            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate

            # формирование случайного набора строк для стохастического градиентного спуска
            sample_rows_idx = list(random.sample(range(X.shape[0]), self.sgd_sample))
            X_sample = X.iloc[sample_rows_idx]
            y_sample = y.iloc[sample_rows_idx]

            # предсказание
            y_pred = np.dot(X_sample, self.weights)
            delta = y_pred - y_sample

            # elastic net loss
            loss = np.mean(delta ** 2) + self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)
            # Расчет градиента на основе сформированного мини-пакета
            gradient_loss = 2 / X_sample.shape[0] * np.dot(X_sample.T, delta) + self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

            # градиентный спуск
            self.weights -= gradient_loss * lr

            metric_score = calculate_metric(y_sample, y_pred, self.metric) if self.metric else loss

            # вывод loss
            if verbose and i % verbose == 0:
                if self.metric:
                    print(f"{i}|loss:{loss}|{self.metric}:{metric_score}")
                else:
                    print(f"{i}|loss:{loss}")
        final_y_pred = np.dot(X, self.weights)
        self.best_score = calculate_metric(y, final_y_pred, self.metric)

    # получение коэффициентов модели
    def get_coef(self):
        return np.array(self.weights[1:])

    # предсказание по заданным параметрам
    def predict(self, X):
        X.insert(0, 'col_0', 1)
        y_pred = np.dot(X, self.weights)
        return np.sum(y_pred, axis=0)

    # последний показатель метрики в обучении
    def get_best_score(self):
        return self.best_score
