import numpy as np
import sqlite3
import dataset_maker as dm
import scipy.special  # для функции сигмоиды


class neuralNetwork:

    # инициализация класса
    # готовим объект перед первым вызовом
    # тут создаем базовые переменные
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # нормальное распределение Гаусса (для более сложной системы весов, см. док
        # аргументы - центр нормального распределения, стандартная девиация (ширина дистрибуции),
        # кортеж параметров (строка, столбец)
        # pow(число, его степень)
        np.random.seed(0)
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))  # !!! seed
        np.random.seed(1)
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))  # !!! seed
        self.lr = learningrate
        # сигмоида
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # метод тренировки
    def train(self, inputs_list, targets_list):
        # превращаем список в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T  # !!!
        hidden_inputs = np.dot(self.wih, inputs)   # получаем матрицу сигналов на вход для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)   # готовый аутпут
        # то же самое для вызодного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # ошибка выходного слоя (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        # обновление весов межку скрытым и выходным слоями (тоже по формуле)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))
        # обновление весов между входным и скрытым слоями
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))
        pass

    # метод непосредственного использования
    def query(self, inputs_list):
        # превращаем список в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)  # получаем матрицу сигналов на вход для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)  # готовый аутпут
        final_inputs = np.dot(self.who, hidden_outputs)
        # то же самое для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# кросс-валидация

def cv_set():
    print('Preparing data, creating CV blocks...')
    conn = sqlite3.connect('characters.db')
    c = conn.cursor()
    c.execute('''SELECT vectorization, 
                 arr_target
                 FROM train
                            ''')
    result = c.fetchall()
    cv_arr = []
    datablock = []
    for i in range(len(result)):  # 875
        if len(datablock) < 124:
            datablock.append((
                np.array([float(x) for x in result[i][0].split()]),
                np.array([float(x) for x in result[i][1].split()])))
        else:
            cv_arr.append(datablock)
            print(str(len(cv_arr)) + ' blocks of 7')
            datablock = []
    return cv_arr



def train(train_data, cycles):
    for step in range(cycles):
        for phrase in train_data:
            input = phrase[0]
            target = phrase[1]
            n.train(input, target)
            pass
        pass


def test(query_data):
    scorecard = []  # 1 - истина, 0 - ложь
    sentence_match = []
    for phrase in query_data:
        output = n.query(phrase[0])
        correct_label = phrase[1]  # это для вычисления доли ошибок
        # формирую вариант выхода, идентичый ожидаемому, чтобы вычислить долю ошибки
        label = []
        for el in output:
            if el > 0.5:
                label.append(0.99)
            else:
                label.append(0.01)
        match = len([1 for i in range(len(label)) if label[i] == correct_label[i]]) / len(label)
        sentence_match.append(match)
        label = ' '.join(map(str, label)) + ' 0.01' * (output_nodes - len(label))
        if label == ' '.join(map(str, correct_label)):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    scorecard = np.array(scorecard)
    accuracy = scorecard.sum() / len(scorecard)
    sentence_match = sum(sentence_match) / len(sentence_match)
    return accuracy, sentence_match



# data = dm.train_set()
#
# cv_score = 0
#
# for i in range(7):
#     # создаем объект класса
#     input_nodes = dm.find_max()[0]
#     hidden_nodes = 400  # экспериментируем
#     output_nodes = dm.find_max()[1]
#
#     learning_rate = 0.1
#
#     n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#
#     epochs = 100  # количество циклов обучения
#
#     query_data = data
#
#     print(n.wih)


input_nodes = 5
hidden_nodes = 5  # экспериментируем
output_nodes = 5
#
learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#