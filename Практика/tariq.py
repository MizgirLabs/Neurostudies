import numpy
import scipy.special  # для функции сигмоиды
import dataset_maker as dm

class neuralNetwork:

    # инициализация класса
    # готовим объект перед первым вызовом
    # тут создаем базовые переменные
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))  # !!! seed
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))  # !!! seed
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = dm.find_max()[0]
hidden_nodes = 150
output_nodes = dm.find_max()[1]

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

epochs = 100

for step in range(epochs):
    for phrase in dm.train_set():
        input = phrase[0]
        target = phrase[1]
        n.train(input, target)
        pass
    pass

scorecard = []
sentence_match = []
for phrase in dm.query_set():
    output = n.query(phrase[0])
    correct_label = phrase[1]
    label = []
    for el in output:
        if el > 0.5:
            label.append(0.99)
        else:
            label.append(0.01)
    match = len([1 for i in range(len(label)) if label[i] == correct_label[i]]) / len(label)
    sentence_match.append(match)
    label = ' '.join(map(str, label)) + ' 0.01' * (output_nodes - len(label))
    print('n:', label, '\n', 'c:', ' '.join(map(str, correct_label)), '\n', '\n')
    if ' '.join(map(str, label)) == ' '.join(map(str, correct_label)):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard = numpy.array(scorecard)
performance = scorecard.sum() / len(scorecard)
sentence_match = sum(sentence_match) / len(sentence_match)
print('Accuracy (amount of full match): ', performance)
print('Sentence match: ', sentence_match * 100, '%')