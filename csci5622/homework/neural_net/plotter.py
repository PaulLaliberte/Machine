import matplotlib.pyplot as plt

from nn import *




def plotter(arcs, regs, neural_net=True):
    f = gzip.open('../data/tinyMNIST.pkl.gz', 'rb')

    title_arcs = ['100-layers', '200-layers', '400-layers', '800-layers']
    title_regs = ['.2-reg', '.1-reg', '.01-reg', '.001-reg', '.00001-reg', '0.0-reg']
    epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    epoch_labels = ['1', '50', '100', '150', '200', '250', '300', '350', '400', '450', '500']

    train, test = cPickle.load(f) 

    train_accuracy = []
    test_accuracy = []

    if neural_net:
        for reg in regs:
            new_net = Network([196, 200, 10])
            new_net.SGD_train(train, epochs=500, eta=.25, lam=reg, verbose=True, test=test)
            train_accuracy.append(new_net.train_acc)
            test_accuracy.append(new_net.test_acc)


    for r in range(len(train_accuracy)):
        plt.plot(epochs, test_accuracy[r], marker='o', label=title_regs[r])

    plt.xticks(epochs, epoch_labels)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Testing Accuracy of Varying Regularizations')
    plt.legend(loc=4)

    plt.show()


if __name__ == "__main__":
                
    list_of_architectures = [[196,100,10], [196,200,10], 
                             [196,400,10], [196,800,10]]

    list_of_regs = [.2, .1, .01, .001, .00001, 0.0]

    plotter(list_of_architectures, list_of_regs, True)

