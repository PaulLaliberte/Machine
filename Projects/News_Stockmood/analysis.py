
#TODO: Find Features

import pandas as pd
import nltk
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

class StockNewsAnalysis():
    
    def __init__(self):
        self.df = None
        self.df_transpose = None
        self.words_list = []
        self.news_stories = defaultdict(dict)
        self.most_common_words = []
        self.featuresets = defaultdict(dict)

    def preprocess(self):
        self.df = pd.read_csv('Combined_News_DJIA.csv')

        #transpose dataset and fill nan with 'NaN' string
        self.df_transpose = self.df.T
        self.df.fillna('NaN', inplace=True)
        self.df_transpose.fillna('NaN', inplace=True)
 
        for i in range(1,26):
            string_Top = 'Top'
            string_Top = ''.join((string_Top, str(i)))

            for j in range(0, len(self.df[string_Top])):
                sub_list = self.df[string_Top][j].split()
                for k in sub_list:
                    self.words_list.append(k)

        
    def frequencey_of_words(self):
        self.words_list = nltk.FreqDist(self.words_list)
        #print(self.words_list.most_common(15))

        for i in self.words_list.most_common(5):
            self.most_common_words.append(i)


    def find_features(self):
        words = set(self.words_list)
        labels = [i for i in self.df_transpose.loc['Label']]

        #use scipy word space vector


    def naive_bayes(self):
        #Need traing and testing data
        #Need to formate data as date_vector = [{dict}, {dict}, ..] and label_vector = labels

        label_vector = [i for i in self.df_transpose.loc['Label']]

            


if __name__ == '__main__':
    n = StockNewsAnalysis()
    n.preprocess()
    n.frequencey_of_words()
    n.find_features()

