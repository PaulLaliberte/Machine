import pandas as pd
import numpy as np
import sklearn

from sklearn.tree import DecisionTreeClassifier

#TODO: Boosting


def processData(df):
    df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    df = df.dropna()

    df['Gender'] = df['Sex'].map({'male' : 0, 'female' : 1}).astype(int)

    df = df.drop(['Sex'], axis=1)

    cols = df.columns.tolist()
    cols = [cols[1]] + cols[0:1] + cols[2:]
    df = df[cols]

    return df.values


if __name__ == "__main__":
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    train_data = processData(df_train)
    test_data = processData(df_test)

    clf = DecisionTreeClassifier(max_depth=1, criterion='entropy')
    clf = clf.fit(train_data[0:, 2:], train_data[0:, 0])

    output = clf.predict(test_data[:,1:])

    results = np.c_[test_data[:,0].astype(int), output.astype(int)]
    df_result = pd.DataFrame(results[:,0:2], columns=['PassengerId', 'Survived'])

    print(df_result)


