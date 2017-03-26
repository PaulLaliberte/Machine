import numpy as np 

kINSP = np.array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = np.array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector w. 
    The vector w should be returned as an Numpy array. 
    """

    w = sum(alpha[i] * y[i] * x[i] for i in range(0, len(x)))
    return w



def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a set of training examples and primal weights, return the indices 
    of all of the support vectors as a set. 
    """

    support = set(i for i in range(0, len(x)) if (np.abs(y[i] * (np.dot(w, x[i]) + b)) - 1.0) <= tolerance)
    return support



def find_slack(x, y, w, b):
    """
    Given a set of training examples and primal weights, return the indices 
    of all examples with nonzero slack as a set.  
    """

    slack = set(i for i in range(0, len(x)) if -1 * (y[i] * (np.dot(w, x[i]) + b) - 1) > 0.0)
    return slack

"""
if __name__ == '__main__':
    print find_support(kSEP[:, 0:2], kSEP[:, 2], np.array([0.2, 0.8]), -.2)
    print weight_vector(kSEP[:, 0:2], kSEP[:, 2], [.12, 0.0, .22, 0, .34, 0, 0])
    print find_slack(kINSP[:, 0:2], kINSP[:, 2], np.array([-.25, .25]), -.25)
    print find_slack(kINSP[:, 0:2], kINSP[:, 2], np.array([0, 2]), -5)
"""
