import numpy as np
import sklearn as sk
import sklearn.linear_model as lm
import matplotlib.pyplot as plt


def get_grid(n, D, x_min, x_max):
    """Генерирует сетку размера n^D x D"""
    xn = np.linspace(x_min, x_max, n)  # массив чисел n чисел от min до max
    xn = np.meshgrid(*([xn] * D))  # матрица из n строк и числами из linspace + та же матрица, только транспонированная
    # сoncatеnate - склеиваем массивы
    # в [xi.reshape((n**D, 1)) for xi in xn] матрица xn, только одна строка - одно число
    return np.concatenate([xi.reshape((n ** D, 1)) for xi in xn], axis=1)
    # в ретурне та же матрица xn, только в два столбика. Первый столбик - первая матрица, второй - вторая


def gen_data(n, D, x_min, x_max, f_target, f_noise):
    """Возвращает аргументы и зашумлённые значения для заданной функции

    Данная функция принимает на вход параметры выборки, которую требуется
    сгенерировать, а так же ссылки на функции, которые должны
    использоваться при генерации.

    n        -- размер одномерной выборки (совпадает с N при D=1)
    D        -- размерность выборки
    f_target -- целевая функция, которую будет аппроксимировать регрессия
    f_noise  -- функция, которая генерирует шум

    Возвращает сгенерированные данные и ответы на этих данных, а так же
    истинные значения функции

    X    -- выборка размера NxD, где N=n^D
    y    -- зашумлённые значения целевой функции
    y_gt -- истинные значения целевой функции"""
    X = get_grid(n, D, x_min, x_max)
    N = X.shape[0]
    y_gt = f_target(X)  # target
    y = y_gt + f_noise(N)
    return X, y, y_gt

def f_noise(N):
    """Обратите внимание, что функция random.normal принимает на вход
    стандартное отклонение, т.е. корень из диспресии"""
    return np.random.normal(loc=2, scale=2, size=N).reshape((N, 1))


def f_target(X):
    return X

N_grid = list(range(10, 1000, 10)) # список N, для которых требуется провести эксперимент

err = []
LRs = []
for N in range(len(N_grid)):
    X, y, y_gt = gen_data(N_grid[N], 1, 0, 3, f_target, f_noise)  # Сгенерированные данные
    part = int(len(X) // 3)
    train_X = X[:part]
    query_X = X[part:]
    train_y = y[:part]
    query_y = y[part:]
    LR = lm.LinearRegression()
    LR.fit(train_X, train_y)
    pred_y = LR.predict(query_X)
    err.append(np.mean((pred_y - query_y)**2))
    LRs.append(LR)
    pass


from sklearn.neighbors import KNeighborsRegressor

err = []
KNNs = []

for N in range(len(N_grid)):
    X, y, y_gt = gen_data(N_grid[N], 1, 0, 3, f_target, f_noise)
    KNN = KNeighborsRegressor(n_neighbors=3)
    KNN.fit(X, y)
    pred = KNN.predict(X)
    err.append(np.mean((pred - y) ** 2))
    KNNs.append(KNN)
    pass  # Place your code here



import sklearn.preprocessing as pp

#def f_target(X):
#    return np.log(0.3*X)

def f_noise(N):
    return np.random.normal(loc=0, scale=1, size=N).reshape((N, 1))

N_grid = np.arange(10, 1000, 10)
X, y, y_gt = gen_data(N_grid[-1], 1, 1, 10, f_target, f_noise)

err_1 = []
LRs_1 = []
for N in range(len(N_grid)):

#    X, y, y_gt = gen_data(N_grid[N], 1, 0, 3, f_target, f_noise)  # Сгенерированные данные
#    part = int(len(X) // 3)
#    print(X)
#    LR = lm.LinearRegression()
#    LR.fit(X, y)
#    LRs_1.append(LR)
#    print(LR)
#    pass  # Place your code here


    # train_X = X[:part]
    # query_X = X[part:]
    # train_y = y[:part]
    # query_y = y[part:]
    # poly = pp.PolynomialFeatures(1)
    # X_new = poly.fit_transform(X)