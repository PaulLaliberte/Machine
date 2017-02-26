from csv import DictReader, DictWriter

import numpy as np
import pandas as pd
import string
import nltk

from movie_data import *
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.stem.snowball import *
from imdb import IMDb
from collections import defaultdict 

from scipy.sparse import csr_matrix

ia = IMDb()

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

#stemmer = PorterStemmer()
stemmer = SnowballStemmer('english', ignore_stopwords=True)

def stem_tokens(tokens, stemmer):
    stemmed = []
    tokens = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    tokens = [s for s in tokens if s]
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def remove_punctuation(sentence):
    punc = re.compile(r'[!?,.:;|0-9]')
    sent = punc.sub('', sentence)

    return sent

class Lemmatize:
    #lemmatize instead of stem
    def __init__(self):
        self.word_net = WordNetLemmatizer()

    def __call__(self, examples):
        return [self.word_net.lemmatize(w) for w in word_tokenize(examples)]


class Feature1(BaseEstimator, TransformerMixin):
    #KEY SPOILER WORDS - may not have as large of an effect as thought
    def __init__(self):
        pass

    def fit(self, examples, y=None):
        return self

    def transform(self, examples, y=None):
        spoilers = ['killed', 'kill', 'died', 'killer', 'dead', "''", 'ends', 
                    'victims', 'victim', 'series', 'finale', 'season', 'back',
                    'scene', 'represented', 'turned', 'had', 'pushed', 'subverted',
                    'worse', 'discovered', 'after', 'death', 'revealed']

        spoilers2 = ['killed', 'death', 'kill', 'ends', 'revealed', 'finale',
                     'killing', 'dead', 'kills', 'final', 'save', 'shot', 'murder',
                     'averted', 'shoot', 'spoiler', 'spoil', 'reveal', 'kissed']

        X = np.zeros((len(examples), 1))

        tokenizer = RegexpTokenizer(r'\w+')

        for i, x in enumerate(examples):
            x = x.lower()
            tokens = tokenizer.tokenize(x)
            for j in spoilers2:
                if j in tokens:
                    X[i,:] = np.array([1])
                else:
                    X[i,:] = np.array([0])
        return csr_matrix(X)

class Feature2(BaseEstimator,  TransformerMixin):
    #GENRE - Get from IMBd
    #scored a little better on cross val...
    #not so much on public score

    def __init__(self, series):
        self.series = series

    def fit(self, examples, y=None):
        return self

    def transform(self, examples, y=None):
        genres = ['Mystery', 'Action', 'Thriller']

        genre_dict = np.load('genredict.npy').item()

        X = np.zeros((len(examples), 1))

        for i, x in enumerate(examples):
            imdb_genres = genre_dict[self.series[i]]

            for g in genres:
                if g in imdb_genres:
                    X[i,:] = np.array([1])
                else:
                    X[i,:] = np.array([0])      
                    
        return csr_matrix(X)
                
class Feature3(BaseEstimator, TransformerMixin):
    #Plot key words
    #not effective - 
    #maybe with an ability to predict the movies one could
    #use plot key words

    def __init__(self, plot_words):
        self.plot_words = plot_words

    def fit(self, examples, y=None):
        return self

    def transform(self, examples, y=None):
        
        X = np.zeros((len(examples), 1))

        for i, x in enumerate(examples):
            words = x.split(" ")
            for i in words:
                if i in self.plot_words:
                     X[i,:] = np.array([1])
                else:
                     X[i,:] = np.array([0])
            X[i,:] = np.array([x.count(word) for word in self.plot_words])

        return csr_matrix(X)

class Feature4(BaseEstimator, TransformerMixin):
    #length

    def __init__(self):
        pass

    def fit(self, examples, y=None):
        return self

    def transform(self, examples, y=None):
        X = np.zeros((len(examples), 1))

        for i, x in enumerate(examples):
            punc = ['', '.', ',', '!', '?']
            x = x.split(" ")
            x = [w for w in x if w not in punc]
            sentence_len = len(x)
            if sentence_len > 40:
                X[i,:] = np.array([1])
            else:
                X[i,:] = np.array([0])

        return csr_matrix(X)

class Feature5(BaseEstimator, TransformerMixin):
    #Count the number of verbs (all forms)

    def __init__(self):
        pass

    def fit(self, examples, y=None):
        return self

    def transform(self, examples, y=None):

        X = np.zeros((len(examples), 2))


        verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        #nouns = ['NN', 'NNS', 'NNP', 'NNPS']


        for num,i in enumerate(examples):
            tag = nltk.pos_tag(nltk.word_tokenize(i))

            counts_verbs = [verbs.count(j[1]) for j in tag]
            #counts_nouns = [nouns.count(j[1]) for j in tag]

            sum_counts_verbs = sum(counts_verbs)
            #sum_counts_nouns = sum(counts_nouns)

            if sum_counts_verbs > 13:
                X[num,:] = 1
            else:
                X[num,:] = 0


        return csr_matrix(X)

class Feature6(BaseEstimator, TransformerMixin):
    #runtime of movie

    def __init__(self, series):
        self.series = series

    def fit(self, examples, y=None):
        return self

    def transform(self, examples, y=None):
        #build dict of runtime/season length

        r = np.load('runtimedict.npy').item()

        X = np.zeros((len(examples), 1))

        for i,x in enumerate(examples):
            length = r[self.series[i]]

            if length != 'unknown':
                length = length[0].split(":")
                if len(length) > 1:
                    length[1].strip('u')
                    length = int(length[1])
                else:
                    length[0].strip('u')
                    length = int(length[0])

                if length >= 60:
                    X[i,:] = np.array([1])
                else:
                    X[i,:] = np.array([0])
            
            else:
                X[i,:] = np.array([0])

        return csr_matrix(X)

            

if __name__ == "__main__":   

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    series_names = []

    for i in train:
        series = i['page']
        series_names.append(series)

    series_names = np.array(series_names)


    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])    

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    x_train = [x[kTEXT_FIELD].lower() for x in train]
    x_test = [x[kTEXT_FIELD].lower() for x in test]

    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,2), 
                                      preprocessor=remove_punctuation)),
            ('feature5', Feature5()),
            ('feature1', Feature2(series_names))
        ])),
        ('classifier',  SGDClassifier(loss='log', penalty='l2', shuffle=True))
        ])


    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)

    o = DictWriter(open("predictions.csv", 'w'), ["Id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['Id'] for x in test], predictions):
        d = {'Id': ii, 'spoiler': labels[pp]}
        o.writerow(d)

    #cross validation scores
    scores = cross_val_score(pipeline, x_train, y_train, cv=5)


    print '\n'
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    print '\n'



    
