import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(18, 16.5)

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


#result = pd.concat([eta, etaa, etaaa, etaaaa, etaaaaa, etaaaaaa], axis=1)

result = pd.concat([tfidf, tfidff, tfidfff, tfidffff, tfidfffff,
                   tfidffffff], axis=1)
plt.figure(); result.plot(); plt.legend(loc='best')
plt.xlabel('Iteration of Update')
plt.ylabel('Accuracy of Test Data')

plt.show()


