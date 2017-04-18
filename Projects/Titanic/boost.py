import numpy as np 
import pandas as pd

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone 

#NOTE: SCORE=.73206

def processData(df, train=True):
    df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].fillna(mean_age)

    fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
    df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)

    df['Gender'] = df['Sex'].map({'male' : 0, 'female' : 1}).astype(int)

    df = df.drop(['Sex'], axis=1)

    if train == True:
        cols = df.columns.tolist()
        cols = [cols[1]] + cols[0:1] + cols[2:]
        df = df[cols]

    return df.values

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=1, criterion='entropy')):        
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        self.boosting_scores = []   #individual boosting scores for each iteration
        
    def fit(self, X_train, y_train):
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

        predicted = np.zeros(len(X))
        for m in range(0, len(self.learners)):
            predicted += self.alpha[m] * self.learners[m].predict(X)
            boost_pred = [np.sign(i) for i in predicted]
            self.boosting_scores.append(boost_pred)

        predicted = [np.sign(i) for i in predicted]

        return predicted


if __name__ == "__main__":
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    train_data = processData(df_train)
    test_data = processData(df_test, False)

    clf = AdaBoost(n_learners=1, base=DecisionTreeClassifier(max_depth=1, criterion="entropy"))
    clf.fit(train_data[0:, 2:], train_data[0:, 0])
    output = np.array(clf.predict(test_data[:,1:]))    

    result = np.c_[test_data[:,0].astype(int), output.astype(int)]
    df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

    df_result.to_csv('titanic_boost.csv', index=False)

