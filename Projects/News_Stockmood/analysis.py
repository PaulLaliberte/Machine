
#TODO: Find Features

import pandas as pd
import nltk
from datetime import date
from dateutil.rrule import rrule, DAILY
from tqdm import tqdm
from collections import defaultdict

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

        start_date = date(2008, 8, 8)
        end_date = date(2016, 7, 1)

        #tqdm for progress bars - must convert to a list to get
        #an estimated complete time
        for dt in tqdm(list(rrule(DAILY, dtstart=start_date, until=end_date))):
            for i in range(1,27):
                for w in self.most_common_words:
                    dict_entry = {}
                    dict_entry[labels[i]] = (w in 
                    self.featuresets[(dt.strftime('%Y-%m-%d'), ''.join(('Top', str(i))), labels[i-1])][w[0]] = (w in words)
           

    def naive_bayes(self):
        #Need traing and testing data
        #Need to formate data as date_vector = [{dict}, {dict}, ..] and label_vector = labels

        label_vector = [i for i in self.df_transpose.loc['Label']]

            


if __name__ == '__main__':
    n = StockNewsAnalysis()
    n.preprocess()
    n.frequencey_of_words()
    n.find_features()

