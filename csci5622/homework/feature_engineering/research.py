import numpy as np
import pandas as pd
import operator
import copy
import nltk
import string
import operator

from csv import DictReader, DictWriter
from collections import defaultdict
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
from imdb import IMDb
from nltk.collocations import *
ia = IMDb()

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

def try1(): 
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])    

    y_train = np.array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    x_train = [x[kTEXT_FIELD] for x in train]

    spoiler_words = defaultdict(int)
    not_ = defaultdict(int)

    for i in range(0, len(x_train)):
        x = x_train[i].split(" ")
        for j in x:
            if y_train[i] == 1:
                if j.lower() not in stopwords.words('english'):
                    spoiler_words[j.lower()] += 1
            elif y_train[i] == 0:
                if j.lower() not in stopwords.words('english'):
                    not_[j.lower()] += 1

    sorted_spoil = sorted(spoiler_words.items(), key=operator.itemgetter(1), reverse=True)[:200]
    sorted_not_ = sorted(not_.items(), key=operator.itemgetter(1), reverse=True)[:200]

    a = [i[0] for i in sorted_spoil]
    b = [i[0] for i in sorted_not_]

    words_only_in_s = []

    for w in a:
        if w not in b:
            words_only_in_s.append(w)

def try2():
    tokenizer = RegexpTokenizer(r'\w+')

    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])    

    y_train = np.array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    x_train = [x[kTEXT_FIELD] for x in train]

    spoiler_words = []

    for i in range(0, len(x_train)):
        if y_train[i] == 1:
            a = x_train[i].lower()
            token = tokenizer.tokenize(a)
            for j in token:
                spoiler_words.append(j)

    en_stop = get_stop_words('en')

    spoiler_words___ = [i for i in spoiler_words if not i in en_stop]

    d = corpora.Dictionary([spoiler_words___])

    
    corpus = [d.doc2bow([text]) for text in spoiler_words___]

    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=d, passes=5)

    print ldamodel.print_topics(num_topics=10, num_words=1)

def try3():
        """
    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    series_names = []

    for i in train:
        series = i['page']
        if series not in series_names:
            series_names.append(series)

    series_names = np.array(series_names)

    genres = defaultdict(lambda: "unknown")

    for i in series_names:
        if genres[i] == "unknown":
            try:
                s_result = ia.search_movie(i)[0]

                ia.update(s_result)

                try:
                    print i
                    g = s_result['genre']
                    genres[i] = g
                except KeyError:
                    genres[i]
            except Exception:
                genres[i]

    genres = dict(genres)

    np.save('genredict.npy', genres)
    

    #read_dict = np.load('genredict.npy').item()
    """

def try4():
        """
    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    series_names = []

    for i in train:
        series = i['page']
        if series not in series_names:
            series_names.append(series)

    series_names = np.array(series_names)

    runtime = defaultdict(lambda: "unknown")

    for i in series_names:
        if runtime[i] == "unknown":
            try:
                s_result = ia.search_movie(i)[0]

                ia.update(s_result)

                try:
                    print i
                    g = s_result['runtime']
                    runtime[i] = g[0].split("::")
                except Exception:
                    runtime[i]
            except Exception:
                runtime[i]

    runtime = dict(runtime)

    np.save('runtimedict.npy', runtime)
    

    r = np.load('runtimedict.npy').item()

    print r
    """

def try5():
    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])    

    y_train = np.array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    x_train = [x[kTEXT_FIELD].lower() for x in train]
    x_test = [x[kTEXT_FIELD].lower() for x in test]

    spoilers = []
    not_spoil = []

    for i in range(len(x_train)):
        if y_train[i] == 1:
            spoilers.append(x_train[i])
        else:
            not_spoil.append(x_train[i])

    spoiler_dict = defaultdict(int)
    not_dict = defaultdict(int)

    tokens_spoil = [''.join(c for c in s if c not in string.punctuation) for s in spoilers]
    tokens_spoil = [s for s in tokens_spoil if s]

    tokens_not = [''.join(c for c in s if c not in string.punctuation) for s in not_spoil]
    tokens_not = [s for s in tokens_not if s]

    for i in range(len(tokens_spoil)):
        word_list = tokens_spoil[i].split(" ")
        for j in word_list:
            if j in nltk.corpus.stopwords.words('english'):
                pass
            else:
                spoiler_dict[j] += 1


    for i in range(len(tokens_not)):
        word_list = tokens_not[i].split(" ")
        for j in word_list:
            if j in nltk.corpus.stopwords.words('english'):
                pass
            else:
                not_dict[j] += 1

def try6():
        # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])    

    y_train = np.array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    x_train = [x[kTEXT_FIELD].lower() for x in train]
    x_test = [x[kTEXT_FIELD].lower() for x in test]

    x = x_train[0]

    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nouns = ['NN', 'NNS', 'NNP', 'NNPS']

    dict_examples = defaultdict(int)

    for i in range(len(x_train)):

        tag = nltk.pos_tag(nltk.word_tokenize(x_train[i]))

        counts = [verbs.count(j[1]) for j in tag]

        sum_counts = sum(counts)

        spoil = 'False'

        if y_train[i] == 0:
            dict_examples[i] = {'spoiler' : spoil, 'verbs' : sum_counts}

    list_dict = sorted(dict_examples.items(), key=operator.itemgetter(1), reverse=True)
    df = pd.DataFrame(list_dict)

    pd.to_pickle(df, 'verbs')


def try7():
    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])    

    y_train = np.array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    x_train = [x[kTEXT_FIELD].lower() for x in train]
    x_test = [x[kTEXT_FIELD].lower() for x in test]

    spoiler_bigrams = defaultdict(int)
    not_bigrams = defaultdict(int)

    for i in range(len(x_train)):
        if y_train[i] == 1:
            tokens = [s for s in x_train[i].split(" ") if s not in stopwords.words("english")]
            tokens = [c for c in tokens if c not in string.punctuation]


            bigrams = BigramCollocationFinder.from_words(tokens)

            for k,v in bigrams.ngram_fd.items():
                spoiler_bigrams[k] += 1
        else:
            tokens = [s for s in x_train[i].split(" ") if s not in stopwords.words("english")]
            tokens = [c for c in tokens if c not in string.punctuation]


            bigrams = BigramCollocationFinder.from_words(tokens)

            for k,v in bigrams.ngram_fd.items():
                not_bigrams[k] += 1



    print sorted(spoiler_bigrams.items(), key=operator.itemgetter(1), reverse=True)[:20], '\n'
    print sorted(not_bigrams.items(), key=operator.itemgetter(1), reverse=True)[:20]







if __name__ == '__main__':
    try6()




        
