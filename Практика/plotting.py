import matplotlib.pyplot as plt

# гарфик качества обучении при каждом новом разбиении на блоки

def ccv(x, y1, y2):
    plt.plot(x, y1)
    plt.title('Accuracy')
    plt.xlabel('ITERATION')
    plt.ylabel('SCORE')
    plt.savefig('./plots/total_accuracy.png', format='png', dpi=100)
    plt.clf()
    plt.plot(x, y2)
    plt.title('Sentence_match')
    plt.xlabel('ITERATION')
    plt.ylabel('SCORE')
    plt.savefig('./plots/total_smatch.png', format='png', dpi=100)
    
# графики по гиперпараметрам

def epochs(acc, sm, x):
    plt.plot(x, acc)
    plt.xlabel('AMOUNT OF TRAINING CYCLES')
    plt.ylabel('ACCURACY')
    plt.savefig('./plots/epochs_accuracy.png', format='png', dpi=100)
    plt.clf()
    plt.plot(x, sm)
    plt.xlabel('AMOUNT OF TRAINING CYCLES')
    plt.ylabel('SENTENCE MATCH')
    plt.savefig('./plots/epochs_smatch.png', format='png', dpi=100)

def hidden_nodes(acc, sm, x):
    plt.plot(x, acc)
    plt.xlabel('AMOUNT OF HIDDEN NODES')
    plt.ylabel('ACCURACY')
    plt.savefig('./plots/hn_accuracy.png', format='png', dpi=100)
    plt.clf()
    plt.plot(x, sm)
    plt.xlabel('AMOUNT OF HIDDEN NODES')
    plt.ylabel('SENTENCE MATCH')
    plt.savefig('./plots/hn_smatch.png', format='png', dpi=100)

def learning_rate(acc, sm, x):
    plt.plot(x, acc)
    plt.xlabel('LEARNING RATE')
    plt.ylabel('ACCURACY')
    plt.savefig('./plots/lr_accuracy.png', format='png', dpi=100)
    plt.clf()
    plt.plot(x, sm)
    plt.xlabel('LEARNING RATE')
    plt.ylabel('SENTENCE MATCH')
    plt.savefig('./plots/lr_smatch.png', format='png', dpi=100)