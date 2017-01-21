"""
    Documentation: http://scikit-learn.org/stable/modules/neighbors.html
"""


import numpy as np

"""
#1.
from sklearn.neighbors import NearestNeighbors

X = np.array([[-1,-1], [-2,-1], [-3,-2], [1,1], [2,1], [3,2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)


print indices, '\n'
print distances, '\n'
print nbrs.kneighbors_graph(X).toarray()
"""

#2.
from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
nbrs = KNeighborsClassifier(n_neighbors=3)
nbrs.fit(X, y)

print nbrs.predict([[1.1]])
print nbrs.predict_proba([[0.9]])
