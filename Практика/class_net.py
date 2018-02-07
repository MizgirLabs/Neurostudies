import numpy as np
import scipy.special as sp  # для функции сигмоиды


class NeuralNetwork:  # это класс

    # инициализация класса
    # эта функция нужна для того, чтобы подготовить объект перед первым вызовом
    # тут создаем базовые переменные
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):  # self - обязательный атрибут метода в питоне
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # нормальное распределение Гаусса (для более сложной системы весов, см. док
        # аргументы - центр нормального распределения, стандартная девиация (ширина дистрибуции),
        # кортеж параметров (строка, столбец)
        # pow(число, его степень)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))  # слой input-hidden
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))  # слой hidden-output
        self.lr = learningrate  # шаг обучения
        # сигмоида
        self.activation_function = lambda x: sp.expit(x)
        pass

    # метод тренировки
    def train(self, inputs_list, targets_list):
        # превращаем список в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T  # !!!
        hidden_inputs = np.dot(self.wih, inputs)  # получаем матрицу сигналов на вход для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)    # готовый аутпут
        # то же самое для вызодного слоя
        final_inputs = np.dot(self.wih, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # ошибка выходного слоя (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибка скрытого слоя - это ошибка выходного слоя, поделенная на веса, перекомбинированные
                                                # на нейроны скрытого слоя (по формуле errors_hidden)
        hidden_errors = np.dot(self.who.T, output_errors)
        # обновление весов межку скрытым и выходным слоями (тоже по формуле)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        # обновление весов между входным и скрытым слоями
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))
        pass

    # метод непосредсивенного использования
    def query(self, inputs_list):
        # превращаем список в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)  # получаем матрицу сигналов на вход для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)  # готовый аутпут
        # то же самое для вызодного слоя
        final_inputs = np.dot(self.wih, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# создаем объект класса
net = NeuralNetwork(3, 3, 3, 0.3)
print(net.query([1.0, 0.5, -1.5]))