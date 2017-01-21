Support Vector Machines
=

Due: 23. September at 11:55pm 

Overview
--------

In this homework you'll explore the primal and dual representations of support
vector machines.

You'll turn in your code on Moodle.  This assignment is worth 25
points.

What you have to do
----

Coding (20pts):

1.  Given a weight vector, implement the *find support* function that returns the indices of the support vectors.
1.  Given a weight vector, implement the *find slack* function that returns the indices of the vectors with nonzero slack.
1.  Given the alpha dual vector, implement the *weight vector* function that returns the corresponding weight vector.

Analysis (5pts):

1.  Use the Sklearn implementation of support vector machines to train a classifier to distinguish 3's from 8's (using the MNIST data from the KNN homework).
1.  Try at least five values of the regularization parameter _C_ and at least two kernels.  Comment on performance for the varying parameters you chose by either testing on a hold-out set or performing cross-validation. 
1.  Give examples of support vectors from each class when using a linear kernel.

Notes
-

- I've provided you a sample driver function that reads in the data and plots training examples.  You will have to add the rest to do the Analysis portion.  Do **NOT** submit your driver file to Moodle. 
- Sklearn's implementation of support vector machines gives a convenient method for extracting support vectors from each class.  Feel free to use that for the analysis portion of the assignment.  


What to turn in
-

1.  Submit your _svm.py_ file
1.  Submit your _analysis.pdf_ file (no more than one page; pictures
    are better than text)


Unit Tests
=

I've provided unit tests based on the example that we worked through in class, plus one additional example that will be added to the Lecture 6 Jupyter notebook.
Make sure it passes all of the unit tests.  However, these tests are not exhaustive; passing the tests will not
guarantee a good grade, you should verify yourself that your code is robust and
correct.


Hints
-

1.  Don't use all of the data, especially at first.  
