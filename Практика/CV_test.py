# Однослойная нерекуррентная нейросеть с кросс-валидацией


import numpy as np
import sqlite3
import dataset_maker as dm
import scipy.special  # для функции сигмоиды
import plotting

# создание класса

class NeuralNetwork:

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
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        np.random.seed(1)
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
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


# поиск оптимальных гиперпараметров
# каждый параметр заменяем на переменную, arr - набор вариантов переменных

# пытаемся подобрать оптимальное число тренировочных циклов

def epochs_optimisation(arr):
    sm = []
    acc = []
    print('Validating training cycles...')
    for i in arr:
        print(i, ' cycles, max: ', max(arr))
        accuracies, sentence_matches, av_accuracy, av_sentence_match = complete_cv(i)
        sm.append(av_sentence_match)
        acc.append(av_accuracy)
    print('Plotting...')
    plotting.epochs(acc, sm, arr)

# подбираем количество нейронов в скрытом слое

def hidden_nodes_optinization(arr):
    sm = []
    acc = []
    print('Validating hidden nodes...')
    for i in arr:
        print(i, ' nodes, max: ', max(arr))
        accuracies, sentence_matches, av_accuracy, av_sentence_match = complete_cv(i)
        sm.append(av_sentence_match)
        acc.append(av_accuracy)
    print('Plotting...')
    plotting.hidden_nodes(acc, sm, arr)

# подбираем шаг обучения

def learning_rate_optinization(arr):
    sm = []
    acc = []
    print('Validating learning rate...')
    for i in arr:
        print(i, ' rate, max: ', max(arr))
        accuracies, sentence_matches, av_accuracy, av_sentence_match = complete_cv(i)
        sm.append(av_sentence_match)
        acc.append(av_accuracy)
    print('Plotting...')
    plotting.learning_rate(acc, sm, arr)


# функция тренировки для вызова в КВ цикле

def learn(n, train_data, cycles):
    for step in range(cycles):
        for phrase in train_data:
            input = phrase[0]
            target = phrase[1]
            n.train(input, target)
            pass
        pass


def test(n, query_data):
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
        label = ' '.join(map(str, label)) + ' 0.01' * (dm.find_max()[1] - len(label))
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


# кросс-валидация, скользящий контроль - обучаю 7 раз на разных блоках, смотрю среднюю оценку

def complete_cv():
    data = dm.train_set()

    cv_lower = 0
    cv_upper = 126


    accuracies = []
    sentence_matches = []

    print('CV is commenced\n')

    for i in range(7):
        print('Perfoming iteration ' + str(i+1) + '/7')
        # создаем объект класса
        input_nodes = dm.find_max()[0]
        hidden_nodes = 550  # экспериментируем
        output_nodes = dm.find_max()[1]

        learning_rate = 0.18

        n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

        epochs = 270  # количество циклов обучения

        query_data = data[cv_lower:cv_upper]
        train_data = data[:cv_lower] + data[cv_upper:]

        print('Training...')
        learn(n, train_data, epochs)
        print('Trained succesfully')
        accuracy, sentence_match = test(n, query_data)
        print('Tested successfully')
        print('Accuracy (amount of full match): ', accuracy)
        print('Sentence match: ', sentence_match * 100, '%\n')
        accuracies.append(accuracy)
        sentence_matches.append(sentence_match)

        cv_lower += 125
        cv_upper += 125

    av_accuracy = sum(accuracies) / len(accuracies)
    av_sentence_match = sum(sentence_matches) / len(sentence_matches)
    return accuracies, sentence_matches, av_accuracy, av_sentence_match


# посмотрим, что получается

def performance():
    accuracies, sentence_matches, av_accuracy, av_sentence_match = complete_cv()
    xes = [x for x in range(1, 8)]
    print('Plotting...')
    plotting.ccv(xes, accuracies, sentence_matches)
    print('Final score:\n Accuracy: ' + str(av_accuracy) + '\n Sentence match: ' + str(av_sentence_match * 100) + ' %\n\n')


if __name__ == '__main__':
    performance()