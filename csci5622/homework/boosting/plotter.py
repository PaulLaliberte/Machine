import matplotlib.pyplot as plt
from boost import *


def plotter_boost(x_limit, y_limit, x_test, y_test, n_learners):
    depths = [0]
    accuracy_PER_boost = []

    for depth in depths:
        #clf = AdaBoost(n_learners=n_learners, base=DecisionTreeClassifier(max_depth=depth, criterion='entropy'))
        clf = AdaBoost(n_learners=n_learners, base=Perceptron()) 
        clf.fit(x_limit, y_limit)
        clf.predict(x_test)
        scores = clf.staged_score(x_test, y_test)

        accuracy_PER_boost.append(scores)

    for depth_acc in accuracy_PER_boost:
        plt.plot(depth_acc)

    plt.margins(0.01)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Iterations')
    plt.title('Boosting Accuracy Per Iteration: Test Data', loc='center')
    plt.show()




if __name__ == '__main__':
    	parser = argparse.ArgumentParser(description='AdaBoost classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	parser.add_argument('--n_learners', type=int, default=50,
                        help="Number of weak learners to use in boosting")
	args = parser.parse_args()

        data = FoursAndNines("../data/mnist.pkl.gz")

        plotter_boost(data.x_train[:args.limit], data.y_train[:args.limit], data.x_test, data.y_test, args.n_learners)


