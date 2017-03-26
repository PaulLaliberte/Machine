import argparse
import numpy as np 
import cPickle, gzip

from sklearn import metrics
from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV


class ThreesAndEights:
    """
    Class to store MNIST data
    """

    def __init__(self, location):


        # You shouldn't have to modify this class, but you can if
        # you'd like.
		

        # Load the dataset
    	f = gzip.open(location, 'rb')

	train_set, valid_set, test_set = cPickle.load(f)

	self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0],:]
	self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0]]

	shuff = np.arange(self.x_train.shape[0])
	np.random.shuffle(shuff)
	self.x_train = self.x_train[shuff,:]
	self.y_train = self.y_train[shuff]

	self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0],:]
	self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0]]
		
	self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0],:]
	self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0]]

	f.close()

        ################################

 
    def linear_classifier(self, limit):
        clf = SVC(kernel='linear', C=1.0)

        
        #Regular classification process
        clf.fit(self.x_train[:limit], self.y_train[:limit])
        predicted = clf.predict(self.x_test)
        expected = self.y_test
        print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted))
        print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
        print '\n'
        return clf.support_
        """

        #Grid Search ---
        grid = GridSearchCV(clf, param_grid=dict(C=np.logspace(-9, 3, 13)), n_jobs=7)
        grid.fit(self.x_train[:limit], self.y_train[:limit])
        print grid.best_params_
        print grid.best_score_
        """


    def poly_classifier(self, limit):
        clf = SVC(kernel='poly', C=1000.0, degree=7)

        
        #Regular classification process
        clf.fit(self.x_train[:limit], self.y_train[:limit])
        predicted = clf.predict(self.x_test)
        expected = self.y_test
        print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected,
                                                                                                     predicted))
        print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
        """

        #Grid Search ---
        grid = GridSearchCV(clf, param_grid=dict(degree=np.logspace(-9, 3, 13), C=np.logspace(-9, 3, 13)), n_jobs=7)
        grid.fit(self.x_train[:limit], self.y_train[:limit])
        print grid.best_params_
        print grid.best_score_
        """


    def rbf_classifier(self, limit):
        clf = SVC(kernel='rbf', C=100, gamma=.0001)

        
        clf.fit(self.x_train[:limit], self.y_train[:limit])
        predicted = clf.predict(self.x_test)
        expected = self.y_test
        print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected,
                                                                                                     predicted))
        print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
        """
        grid = GridSearchCV(clf, param_grid=dict(gamma=np.logspace(-9, -3, 13), C=np.logspace(-9, 3, 13)), n_jobs=7)
        grid.fit(self.x_train[:limit], self.y_train[:limit])
        print grid.best_params_
        print grid.best_score_
        """


        


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

if __name__ == "__main__":

        parser = argparse.ArgumentParser(description='SVM classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	args = parser.parse_args()
    
	data = ThreesAndEights("../data/mnist.pkl.gz")

        list_of_svs = data.linear_classifier(args.limit)


        #print "\n"
        #data.poly_classifier(args.limit)
        #print "\n"
        #data.rbf_classifier(args.limit)

	# -----------------------------------
	# Plotting Examples 
	# -----------------------------------

	#Display in on screen  
	mnist_digit_show(data.x_train[list_of_svs[0]])
        mnist_digit_show(data.x_train[list_of_svs[1]])
        mnist_digit_show(data.x_train[list_of_svs[2]])
        mnist_digit_show(data.x_train[list_of_svs[len(list_of_svs) - 1]])
        mnist_digit_show(data.x_train[list_of_svs[len(list_of_svs) - 2]])
        mnist_digit_show(data.x_train[list_of_svs[len(list_of_svs) - 3]])









