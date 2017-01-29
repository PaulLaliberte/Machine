

import numpy as np
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


def read_and_clean_data(fname, remove_stops=True):

    with open('stopwords.txt', 'r') as f:
        stops = [line.rstrip('\n') for line in f]

    with open(fname, 'rU') as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')
        labels = []; text = []
        for ii, row in enumerate(reader):
            labels.append(int(row[0]))
            words = row[1].lower().split()
            if words[0][0] == 'b' and (words[0][1] == '"' or words[0][1] == "'"):
                words[0] = words[0][2:]
            words = [w for w in words if not w in stops] if remove_stops else words
            text.append(' '.join(words))

    return text, labels

#example of training data and their classification
text_train, labels_train = read_and_clean_data('cDJIA_train.tsv', remove_stops=True)
text_test, labels_test = read_and_clean_data('cDJIA_test.tsv', remove_stops=True)

tvec = TfidfVectorizer()

y_train = np.array(labels_train)
y_test = np.array(labels_test)
X_tf_train = tvec.fit_transform(text_train)
X_tf_test = tvec.transform(text_test)

tfLR = LogisticRegression()
tfLR.fit(X_tf_train, y_train)
pred_tfLR = tfLR.predict(X_tf_test)

print "Logistic Regression accuracy with tf-idf: ", accuracy_score(y_test, pred_tfLR)
