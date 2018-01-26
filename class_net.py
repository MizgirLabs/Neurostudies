import numpy as np
import scipy.special as sp  # для функции сигмоиды


class NeuralNetwork:  # это класс

    # инициализация класса
    # эта функция нужна для того, чтобы подготовить объект перед первым вызовом
    # тут создаем базовые переменные
    def __init__(self, inputnodes, hiddendnodes, outputnodes, learningrate):  # self - обязательный атрибут метода в питоне
        self.inodes = inputnodes
        self.hnodes = hiddendnodes
        self.onodes = outputnodes

        self.lr = learningrate

        # нормальное распределение Гаусса (для более сложной системы весов, см. док
        # аргументы - центр нормального распределения, стандартная девиация (ширина дистрибуции),
        # кортеж параметров (строка, столбец)
        # pow(число, его степень)
        self.wih = np.random.normal(pow(self.hnodes, -0.5), (self.hnodes, self.inodes))  # слой input-hidden
        self.who = np.random.normal(pow(self.оnodes, -0.5), (self.onodes, self.hnodes))  # слой hidden-output

        # сигмоида
        self.activation_func = lambda x: sp.expit(x)
        pass

    # метод тренировки
    def train(self):
        pass

    # метод непосредсивенного использования
    def query(self):
        hidden_inputs = np.dot(self.wih, inputs)  # получаем матрицу сигналов на вход для скрытого слоя

        pass


# создаем объект класса
net = NeuralNetwork(3, 3, 3, 0.3)
