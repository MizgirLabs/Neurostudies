import numpy as np
import scipy.special  # для функции сигмоиды
import dataset_maker as dm


class NeuralNetwork:  # это класс

    # инициализация класса
    # эта функция нужна для того, чтобы подготовить объект перед первым вызовом
    # тут создаем базовые переменные
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):  # self - обязательный атрибут метода в питоне
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        # нормальное распределение Гаусса (для более сложной системы весов, см. док
        # аргументы - центр нормального распределения, стандартная девиация (ширина дистрибуции),
        # кортеж параметров (строка, столбец)
        # pow(число, его степень)
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))  # слой input-hidden
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))  # слой hidden-output
        self.lr = learning_rate  # шаг обучения
        # сигмоида
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # метод тренировки
    def train(self, inputs_list, targets_list):
        # превращаем список в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T  # !!!
        hidden_inputs = np.dot(self.wih, inputs)  # получаем матрицу сигналов на вход для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)    # готовый аутпут
        # то же самое для вызодного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
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
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# создаем объект класса
i_nodes = 3
h_nodes = 3  # экспериментируем
o_nodes = 3
learn_rate = 3

net = NeuralNetwork(i_nodes, h_nodes, o_nodes, learn_rate)

x = net.query(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))

# тренируем
# epochs = 1 # количество циклов обучения
#
# for step in range(epochs):
#     for phrase in dm.train_set():
#         input = phrase[0]
#         target = phrase[1]
#         net.train(input, target)
#         pass
#     pass
#
# scorecard = [] # 1 - истина, 0 - ложь
# for phrase in dm.query_set():
#     output = net.query(phrase[0])
#     correct_label = phrase[1] # это для вычисления доли ошибок
#     # формирую вариант выхода, идентичый ошидаемому, чтобы вычислить долю ошибки
#     label = []
#     for el in output:
#         if el > 0.5:
#             label.append(0.99)
#         else:
#             label.append(0.01)
#     if label == correct_label:
#         scorecard.append(1)
#     else:
#         scorecard.append(0)
#         pass
#     pass
#
# scorecard = np.array(scorecard)
# performance = scorecard.sum() / len(scorecard)
# print('Доля ошибок: ', performance)


# надо чтобы массив на вход был всегда одинакового размера (добивать нулями?)
# тогда и таргет тоже поменять