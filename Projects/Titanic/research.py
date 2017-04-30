import matplotlib.pyplot as plt

import pandas as pd
import numpy as np




if __name__ == "__main__":
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    df_train['Age'] = df_train['Age'].dropna()



    
