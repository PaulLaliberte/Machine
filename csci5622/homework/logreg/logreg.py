import random
import argparse
import numpy as np
import copy
import operator
#import pandas as pd
#import matplotlib.pyplot as plt

from numpy import zeros, sign 
from math import exp, log
from collections import defaultdict

kSEED = 1735
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)

#for test file (warning divide by zero - line 142). A zero isn't possible, only in test. 
np.seterr(divide='ignore')


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * sign(score)

    return 1.0 / (1.0 + exp(-score))


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {}
        self.df = zeros(len(vocab))
        self.y = label
        self.x = zeros(len(vocab))

        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1

        #NOTE: ec2, df dict... need df != None for test
        if df != None:
            for word, count in [x.split(":") for x in words]:
                if word in vocab:
                    self.df[vocab.index(word)] = df[vocab.index(word)] 


class LogReg:
    def __init__(self, num_features, lam, eta=lambda x: x):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param lam: Regularization parameter
        :param eta: A function that takes the iteration as an argument (the default is a constant value)
        """
        
        self.w = zeros(num_features)
        self.lam = lam
        self.eta = eta
        self.last_update = defaultdict(int)
        self.best_predict = None
        self.tf_idf = defaultdict(float)

        self.last_update['number_at'] = 0

        assert self.lam>= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy
        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ex in examples:
            p = sigmoid(self.w.dot(ex.x))
            if ex.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ex.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, iteration, use_tfidf=False, learning_rate=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.
        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """

        """
        #unregularized
        eta = self.eta(iteration)
        y_i = train_example.y
        sigm = sigmoid(np.dot(self.w, train_example.x))
        w_k = self.w + eta * (y_i - sigm) * train_example.x
        self.w = w_k
        """
        
        #regularized
        #NOTE: For exponent update, add an extra +1, to take into account the current
        #      regularization too

        df = train_example.df

        words = [t for idf,t in train_example.nonzero.items()]
        total_count = len(words)
        tf = {t : 0 for idf,t in train_example.nonzero.items()}
        #tf_extracredit = {t : 0 for idf,t in train_example.nonzero.items()}

        for i in words:
            tf[i] += 1
            #tf_extracredit[i] += 1

        for t,n in tf.items():
            tf[t] = float(n) / total_count
            
        for ind,t in train_example.nonzero.items():
            df_count = df[ind]

            #total number of docs: 1059 + 133
            df[ind] = np.log(1192 / (df_count))

        for ind,t in train_example.nonzero.items():
            df[ind] = df[ind] * tf[t]

            """
            #keep track of best words for analysis
            self.tf_idf[t] += df[ind] 
            """

        assert len(train_example.x) == len(df)

        if use_tfidf == False:
            df = train_example.x
        else:
            for i in df:
                #tf-idf must be positive
                assert i >= 0.0

        eta = self.eta(iteration)

        if learning_rate == True:
            eta = eta_schedule(iteration)

        y_i = train_example.y
        sigm = sigmoid(np.dot(self.w, df))

        indices_to_update = np.array([], dtype=np.int64)
        ind_counter = 0

        for i in df:
            if i != 0:
                indices_to_update = np.insert(indices_to_update, 
                                                  len(indices_to_update), ind_counter)

            ind_counter += 1

        for i in indices_to_update:
            self.w[i] = self.w[i] + eta * (y_i - sigm) * df[i]

        shrink_fact = 1 - 2*eta*self.lam

        for i in indices_to_update:
            if i != 0:
                if self.last_update[i] != self.last_update['number_at']:
                    exponent = self.last_update['number_at'] - self.last_update[i] + 1
                    self.w[i] = self.w[i] * (shrink_fact ** exponent)
                    self.last_update[i] = self.last_update['number_at'] 
                else:
                    self.w[i] = self.w[i] * shrink_fact

        length_of_vec = len(self.w)
        ind_counter = 1

        while ind_counter <= length_of_vec:
            if ind_counter in indices_to_update:
                self.last_update[ind_counter] += 1

            ind_counter += 1

        self.last_update['number_at'] += 1

        return self.w
    
def eta_schedule(iteration):
    #1/i increase

    new_eta = args.eta / (1. + (100 * iteration) / (args.passes*1192))

    return new_eta

def read_dataset(positive, negative, vocab, test_proportion=0.1):
    """
    Reads in a text dataset with a given vocabulary
    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """

    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data 
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab, df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lam", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--eta", help="Initial SG learning rate",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/autos_motorcycles/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/autos_motorcycles/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/autos_motorcycles/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)
    argparser.add_argument("--tfidf", help="Use tf-idf", type=str, default='False', required=False)
    argparser.add_argument("--update", help="Use updated learn rate", type=str, default='False', required=False)

    args = argparser.parse_args()
    train, test, vocab, df = read_dataset(args.positive, args.negative, args.vocab)

    """
    #tf-idf
    tfidf_auto = {k : None for k in vocab}
    tfidf_motor = copy.deepcopy(tfidf_auto)
    """

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.lam, lambda x: args.eta)

    """
    #plot arrays
    plot_update = []
    plot_accuracy = []
    tf_auto_extracredit = {i : 0 for i in vocab}
    tf_cycle_extracredit = {i : 0 for i in vocab}
    """
    
    # Iterations
    iteration = 0

    #covert to bool
    use_tfidf = False
    update_learn = False

    if args.tfidf == 'True':
        use_tfidf = True

    if args.update == 'True':
        update_learn = True

    for pp in xrange(args.passes):
        random.shuffle(train)
        for ex in train:
            lr.sg_update(ex, iteration, use_tfidf, update_learn)
            """
            if ex.y == 1:
                for ind,t in ex.nonzero.items():
                    tf_auto_extracredit[t] += tf[t]
            
                    try:
                        tfidf_auto[t] += tfidf[ind]
                    except TypeError:
                        tfidf_auto[t] = tfidf[ind]
            
            elif ex.y == 0:
                for ind,t in ex.nonzero.items():
                    tf_cycle_extracredit[t] += tf[t]
                    try:
                        tfidf_motor[t] += tfidf[ind]
                    except TypeError:
                        tfidf_motor[t] = tfidf[ind]
            """
            if iteration % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                
                #plot_accuracy.append(ho_acc)
                #plot_update.append(iteration)
                
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (iteration, train_lp, ho_lp, train_acc, ho_acc))

            iteration += 1


"""
    #dataframes
    plot_eta = pd.DataFrame(plot_accuracy, plot_update, columns=['ulr: n, tfidf: y'])
    plot_eta.to_pickle('learn_rate4')


    tf_auto = dict(sorted(tf_auto_extracredit.items(), key=operator.itemgetter(1), reverse=True)[:20])
    tf_cycle = dict(sorted(tf_cycle_extracredit.items(), key=operator.itemgetter(1), reverse=True)[:20])
    df_auto = pd.DataFrame(tf_auto.items(), columns=['term', 'tf'])
    df_cycle = pd.DataFrame(tf_cycle.items(), columns=['term', 'tf'])

    predict_neg = { k : v for k,v in tfidf_auto.items() if v != None}
    predict_pos = { k : v for k,v in tfidf_motor.items() if v != None}

    best_neg = dict(sorted(predict_neg.items(), key=operator.itemgetter(1))[:20])
    best_pos = dict(sorted(predict_pos.items(), key=operator.itemgetter(1))[:20])
    worst_neg = dict(sorted(predict_neg.items(), key=operator.itemgetter(1), reverse=True)[:20])
    worst_pos = dict(sorted(predict_pos.items(), key=operator.itemgetter(1), reverse=True)[:20])

    df_best_neg = pd.DataFrame(best_neg.items(), columns=['term', 'tfidf (sum)'])
    df_worst_neg = pd.DataFrame(worst_neg.items(), columns=['term', 'tfidf (sum)'])
    df_best_pos = pd.DataFrame(best_pos.items(), columns=['term', 'tfidf (sum)'])
    df_worst_pos = pd.DataFrame(worst_pos.items(), columns=['term', 'tfidf (sum)'])

    df_best_neg.to_pickle('worst_neg')
    df_worst_neg.to_pickle('best_neg')
    df_best_pos.to_pickle('worst_pos')
    df_worst_pos.to_pickle('best_pos')

    df_auto.to_pickle('tf_auto')
    df_cycle.to_pickle('tf_cycle')
"""    
