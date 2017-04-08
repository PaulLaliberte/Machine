import argparse
import numpy as np 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone 
import matplotlib.pyplot as plt

np.random.seed(1234)

"""
X = np.array([
[6,9.5],
[4,8.5],
[9,8.75],
[8,8.0],
[3,7],
[1,6.5],
[5,6.5],
[1.5,2.5],
[2,1],
[9,2],
])
y = np.array([1,1,-1,1,-1,1,-1,1,-1,-1])
"""


class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set 
        train_set, valid_set, test_set = cPickle.load(f)

        # Extract only 4's and 9's for training set 
        self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
        self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]
        self.y_train = np.array([1 if y == 9 else -1 for y in self.y_train])
        
        # Shuffle the training data 
        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        # Extract only 4's and 9's for validation set 
        self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
        self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]
        self.y_valid = np.array([1 if y == 9 else -1 for y in self.y_valid])
        
        # Extract only 4's and 9's for test set 
        self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
        self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]
        self.y_test = np.array([1 if y == 9 else -1 for y in self.y_test])
        
        f.close()

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=1, criterion='entropy')):
        """
        Create a new adaboost classifier.
        
        Args:
            n_learners (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner 

        Attributes:
            base (estimator): Your general weak learner 
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners. 
            learners (list): List of weak learner instances. 
        """
        
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        self.boosting_scores = []   #individual boosting scores for each iteration
        
    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alphas and learners. 
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """

        K, _ = X_train.shape
        w = np.ones(K) / K

        for m in range(0, self.n_learners):
            learner = clone(self.base)
            learner.fit(X_train, y_train, sample_weight=w)

            pred = learner.predict(X_train)

            err_m = w.dot(pred != y_train) / sum(w)

            self.alpha[m] = .5 * (np.log((1 - err_m) / err_m))

            w = w * np.exp(-self.alpha[m] * y_train * pred)

            self.learners.append(learner)

    def predict(self, X):
        """
        Adaboost prediction for new data X.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns: 
            [n_samples] ndarray of predicted labels {-1,1}
        """

        predicted = np.zeros(len(X))
        for m in range(0, len(self.learners)):
            predicted += self.alpha[m] * self.learners[m].predict(X)
            boost_pred = [np.sign(i) for i in predicted]
            self.boosting_scores.append(boost_pred)

        predicted = [np.sign(i) for i in predicted]

        return predicted
    
    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.  
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            Prediction accuracy (between 0.0 and 1.0).
        """

        predicted = self.predict(X)
        total = len(y)
        correct = 0.0

        for i in range(0, len(predicted)):
            if predicted[i] == y[i]:
                correct += 1.0

        return correct / total
    
    def staged_score(self, X, y):
        """
        Computes the ensemble score after each iteration of boosting 
        for monitoring purposes, such as to determine the score on a 
        test set after each boost.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            [n_learners] ndarray of scores 
        """

        staged_scores = []
        temp_learners = self.learners

        for i in range(0, len(self.learners)):
            self.learners = temp_learners[:i+1]
            staged_scores.append(self.score(X,y))

        return staged_scores

def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
	    plt.savefig(outname)
	else:
	    plt.show()

"""
if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='AdaBoost classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	parser.add_argument('--n_learners', type=int, default=50,
                        help="Number of weak learners to use in boosting")
	args = parser.parse_args()

	data = FoursAndNines("../data/mnist.pkl.gz")

    # An example of how your classifier might be called 
	clf = AdaBoost(n_learners=400, base=DecisionTreeClassifier(max_depth=1, criterion="entropy"))
        clf.fit(data.x_train[:args.limit], data.y_train[:args.limit])
        predicted = clf.predict(data.x_test)
        expected = data.y_test

        print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected,
                                                                                                     predicted))

        print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)

"""
