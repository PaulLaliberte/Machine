import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(18, 22)

eta = pd.read_pickle('eta1')
etaa = pd.read_pickle('eta2')
etaaa = pd.read_pickle('eta3')
etaaaa = pd.read_pickle('eta4')
etaaaaa = pd.read_pickle('eta5')
etaaaaaa = pd.read_pickle('eta6')

tfidf = pd.read_pickle('tfidf1')
tfidff = pd.read_pickle('tfidf2')
tfidfff = pd.read_pickle('tfidf3')
tfidffff = pd.read_pickle('tfidf4')
tfidfffff = pd.read_pickle('tfidf5')
tfidffffff = pd.read_pickle('tfidf6')

lr1 = pd.read_pickle('learn_rate1')
lr2 = pd.read_pickle('learn_rate2')
lr3 = pd.read_pickle('learn_rate3')
lr4 = pd.read_pickle('learn_rate4')


#result = pd.concat([eta, etaa, etaaa, etaaaa, etaaaaa, etaaaaaa], axis=1)

result = pd.concat([lr1, lr2, lr3, lr4], axis=1)
plt.figure(); result.plot(); plt.legend(loc='best')
plt.xlabel('Iteration of Update')
plt.ylabel('Accuracy of Test Data')

plt.show()


