import sklearn

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

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


if __name__ == "__main__":
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    train_data = processData(df_train)
    test_data = processData(df_test, False)
    

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_data[0:, 2:], train_data[0:, 0])

    output = clf.predict(test_data[:,1:])

    print(output)

    result = np.c_[test_data[:,0].astype(int), output.astype(int)]
    df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

    #df_result.to_csv('titanic_rforest.csv', index=False)

