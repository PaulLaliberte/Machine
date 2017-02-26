import pandas as pd
import numpy as np

class Movie_Data:
    
    def __init__(self):
        self.dataset = pd.DataFrame.from_csv('movie_metadata.csv')
        self.genres = self.dataset['genres']
        self.movie_titles = None
        self.length = None
        self.plot_key_words = self.dataset['plot_keywords']


    def clean_keywords(self):
        self.plot_key_words = self.plot_key_words.dropna()
        self.plot_key_words = self.plot_key_words.as_matrix()

        new_arr = []
        for i in self.plot_key_words:
            words = i.split("|")
            for j in words:
                if j not in new_arr:
                    new_arr.append(j)

        self.plot_key_words = np.array(new_arr)
        self.plot_key_words = self.plot_key_words[:1000]


