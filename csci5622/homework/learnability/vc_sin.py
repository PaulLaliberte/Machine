from math import pi, sin
import argparse

kSIMPLE_TRAIN = [(1, False), (2, True), (4, False), (5, True), (13, False),
                 (14, True), (19, False)]


class SinClassifier:
    """
    A binary classifier that is parameterized a single float 
    """

    def __init__(self, w):
        """
        Create a new classifier parameterized by w

        Args:
          w: The parameter w in the sin function (a real number)
        """
        assert isinstance(w, float)
        self.w = w

    def __call__(self, k):
        """
        Returns the raw output of the classifier.  The sign of this value is the
        final prediction.

        Args:
          k: The exponent in x = 2**(-k) (an integer)
        """

        return sin(self.w * 2 ** (-k))

    def classify(self, k):
        """

        Classifies an integer exponent based on whether the sign of \sin(w * 2^{-k})
        is >= 0.  If it is, the classifier returns True.  Otherwise, false.

        Args:
          k: The exponent in x = 2**(-k) (an integer)
        """
        assert isinstance(k, int), "Object to be classified must be an integer"

        if self(k) >= 0:
            return True
        else:
            return False


def train_sin_classifier(data):
    """
    Compute the correct parameter w of a classifier to prefectly classify the
    data and return the corresponding classifier object

    Args:
      data: A list of tuples; first coordinate is k (integers), second is y (+1/-1)
    """

    assert all(isinstance(k[0], int) and k >= 0 for k in data), \
        "All training inputs must be integers"
    assert all(isinstance(k[1], bool) for k in data), \
        "All labels must be True / False"

    arr = []
    for l in data:
        binary_class = 1
        if l[1] == False:
            binary_class = -1

        arr.append(((1. - binary_class) * (2**l[0]))/2.)

    w = pi * (1 + sum(arr))

    return SinClassifier(w)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cs", help="Example: cannot shatter 4 points",
                           type=str, default=False, required=False)

    args = argparser.parse_args()

    classifier = train_sin_classifier(kSIMPLE_TRAIN)
    for kk, yy in kSIMPLE_TRAIN:
        print(kk, yy, classifier(kk), classifier.classify(kk))

    if args.cs == 'yes' or args.cs == 'Yes' or args.cs == 'y' or args.cs == 'Y':
        print '\n'

        cannot_shatter = [(1, False), (2, False), (3, True), (4, False)]

        print "Example of not begin able to shatter 4 points of equal distance and the following labeling: %s" % cannot_shatter

        print '\n'

        for kk, yy in cannot_shatter:
            print(kk, yy, classifier(kk), classifier.classify(kk))


