Logistic Regression
=

Due: 10. February by 11:55pm

Overview
--------

In this homework you'll implement a stochastic gradient ascent for
logistic regression and you'll apply it to the task of determining
whether documents are talking about hockey or baseball.

![Hockey and Baseball: Are they really that different?](baseball_hockey.jpg "Two sports I know nothing about")

This will be slightly more difficult than the last homework (the
difficulty will slowly ramp upward).  You should not use any libraries that implement any of the functionality of logistic regression
for this assignment; logistic regression is implemented in scikit
learn, but you should do everything by hand now.  You'll be able to
use library implementations of logistic regression in the future. 

You'll turn in your code and analysis on Moodle.  This assignment is worth 30
points.

What you have to do
----

Coding (20 points):

1. Understand how the code is creating feature vectors (this will help you code the solution and to do the later analysis).  You don't actually need to write any code for this, however.
2. Modify the _sg update_ function to perform non-regularized updates.
3. After that works, modify the _sg update_ function to perform regularized updates. 
4. You'll likely need to write some code to get the best/worst features (for the analysis portion).

**NOTES**: 
- You should not regularize the bias term. 
- You should implement Lazy Sparse Regularization and only update non-zero features. 

<!---(See discussion [here](https://nbviewer.jupyter.org/url/grandmaster.colorado.edu/~cketelsen/files/csci5622/notebooks/lesson04/lesson04NBKAnswers.ipynb?flush_cache=true)) -->

Analysis (10 points):

1. How did the learning rate affect the convergence of your SGA implementation?
2. What was your stopping criterion and how many passes over the data did you need to complete before stopping?
3. What words are the best predictors of each class?  How (mathematically) did you find them?
4. What words are the poorest predictors of classes?  How (mathematically) did you find them?

Extra credit:

1. (max 2pts) Use a schedule to update the learning rate.
    - Modify the eta_schedule function 
    - Pass it into the LogReg constructor 
    - Support it in your _sg update_
    - Show the effect in your analysis document
1.  (max 2pts) Use document frequency (provided in the vocabulary file) to modify the feature values to [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).
    - Modify the Example to store the df vector
    - With the appropriate flag, use the df and x vectors to implement tf-idf in the update
    - Show the effect in your analysis document

Caution: When implementing extra credit, make sure your implementation of the
regular algorithms doesn't change.

What to turn in
-

1. Submit your _logreg.py_ file (include your name at the top of the source)
1. Submit your _analysis.pdf_ file
    - no more than one page
    - pictures are better than text
    - include your name at the top of the PDF

Unit Tests
=

I've provided unit tests based on the example that we worked through
in class.  Before running your code on read data, make sure it passes
all of the unit tests.

```
MacBook-Air:logreg cketelsen$ python tests.py 
[ 0.  0.  0.  0.  0.]
[ 1.  4.  3.  1.  0.]
F[ 0.  0.  0.  0.  0.]
[ 1.  4.  3.  1.  0.]
F
======================================================================
FAIL: test_reg (__main__.TestLogReg)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 37, in test_reg
    self.assertAlmostEqual(w[0], .5)
AssertionError: 0.0 != 0.5 within 7 places

======================================================================
FAIL: test_unreg (__main__.TestLogReg)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests.py", line 18, in test_unreg
    self.assertAlmostEqual(w[0], .5)
AssertionError: 0.0 != 0.5 within 7 places

----------------------------------------------------------------------
Ran 2 tests in 0.005s

FAILED (failures=2)
```

Example
-

This is an example of what your runs should look like:
```
MacBook-Air:logreg cketelsen$ python logreg.py
Read in 1064 train and 133 test
Update 1    TP -871.745612  HP -106.044691  TA 0.492481 HA 0.548872
Update 6    TP -1029.731220 HP -140.270909  TA 0.509398 HA 0.436090
Update 11   TP -720.464529  HP -102.363007  TA 0.562030 HA 0.548872
Update 16   TP -1803.939194 HP -203.081063  TA 0.516917 HA 0.548872
Update 21   TP -1745.695111 HP -204.262203  TA 0.539474 HA 0.563910
Update 26   TP -887.148911  HP -111.089238  TA 0.619361 HA 0.669173
Update 31   TP -989.056993  HP -122.695979  TA 0.608083 HA 0.631579
Update 36   TP -1056.983985 HP -159.913546  TA 0.621241 HA 0.571429
Update 41   TP -728.585251  HP -108.873585  TA 0.693609 HA 0.639098
Update 46   TP -834.679008  HP -129.161203  TA 0.669173 HA 0.616541
Update 51   TP -640.657074  HP -78.602569   TA 0.709586 HA 0.669173
Update 56   TP -694.108563  HP -82.211060   TA 0.697368 HA 0.676692
Update 61   TP -607.631992  HP -75.804210   TA 0.717105 HA 0.714286
Update 66   TP -641.874781  HP -78.314354   TA 0.719925 HA 0.684211
Update 71   TP -603.978505  HP -91.005892   TA 0.740602 HA 0.706767
Update 76   TP -551.585853  HP -84.748741   TA 0.770677 HA 0.699248
Update 81   TP -528.865679  HP -77.795893   TA 0.780075 HA 0.759398
Update 86   TP -541.798281  HP -77.471082   TA 0.770677 HA 0.751880
Update 91   TP -539.051418  HP -76.902155   TA 0.774436 HA 0.766917
Update 96   TP -515.521591  HP -81.688677   TA 0.790414 HA 0.744361
Update 101  TP -519.139668  HP -84.574911   TA 0.784774 HA 0.721805
Update 106  TP -538.389433  HP -89.117392   TA 0.781015 HA 0.714286
Update 111  TP -563.132373  HP -96.642420   TA 0.783835 HA 0.706767
Update 116  TP -535.766264  HP -66.476999   TA 0.772556 HA 0.774436
Update 121  TP -549.221281  HP -66.670172   TA 0.761278 HA 0.766917
Update 126  TP -517.987775  HP -64.031395   TA 0.781015 HA 0.804511
Update 131  TP -512.455415  HP -66.506449   TA 0.788534 HA 0.804511
Update 136  TP -569.234532  HP -64.432343   TA 0.765038 HA 0.804511
Update 141  TP -524.615022  HP -60.954919   TA 0.781015 HA 0.812030
Update 146  TP -512.349266  HP -61.342132   TA 0.793233 HA 0.812030
Update 151  TP -491.483515  HP -58.549736   TA 0.806391 HA 0.834586
Update 156  TP -480.108793  HP -61.472416   TA 0.810150 HA 0.819549
Update 161  TP -477.257234  HP -60.227001   TA 0.809211 HA 0.819549
Update 166  TP -472.642585  HP -62.027318   TA 0.816729 HA 0.819549
Update 171  TP -474.482774  HP -64.359936   TA 0.818609 HA 0.819549
Update 176  TP -451.996553  HP -57.784959   TA 0.830827 HA 0.842105
Update 181  TP -439.751698  HP -61.027691   TA 0.837406 HA 0.804511
Update 186  TP -424.819192  HP -58.440725   TA 0.841165 HA 0.812030
Update 191  TP -434.136991  HP -67.736295   TA 0.844925 HA 0.812030
Update 196  TP -522.413963  HP -88.972162   TA 0.793233 HA 0.699248
Update 201  TP -534.133285  HP -91.248178   TA 0.795113 HA 0.684211
Update 206  TP -511.820573  HP -87.218096   TA 0.796992 HA 0.691729
Update 211  TP -456.740090  HP -75.767403   TA 0.831767 HA 0.774436
Update 216  TP -425.847887  HP -70.295784   TA 0.847744 HA 0.781955
Update 221  TP -431.562757  HP -71.563389   TA 0.850564 HA 0.766917
Update 226  TP -416.082289  HP -68.506145   TA 0.857143 HA 0.789474
Update 231  TP -589.901996  HP -66.418090   TA 0.784774 HA 0.736842
Update 236  TP -422.335295  HP -69.338239   TA 0.849624 HA 0.789474
Update 241  TP -467.365950  HP -79.999377   TA 0.829887 HA 0.729323
Update 246  TP -427.340122  HP -69.815172   TA 0.839286 HA 0.796992
Update 251  TP -471.678332  HP -58.456457   TA 0.831767 HA 0.796992
Update 256  TP -461.588325  HP -57.835168   TA 0.834586 HA 0.812030
Update 261  TP -460.403871  HP -57.695754   TA 0.835526 HA 0.804511
Update 266  TP -394.833081  HP -57.631523   TA 0.860902 HA 0.819549
Update 271  TP -396.731887  HP -57.364974   TA 0.856203 HA 0.812030
Update 276  TP -395.490214  HP -53.762357   TA 0.854323 HA 0.819549
Update 281  TP -372.838675  HP -56.139944   TA 0.870301 HA 0.796992
Update 286  TP -375.048914  HP -57.890215   TA 0.873120 HA 0.827068
Update 291  TP -366.166747  HP -55.161124   TA 0.877820 HA 0.834586
Update 296  TP -365.272303  HP -55.342908   TA 0.879699 HA 0.842105
Update 301  TP -365.527768  HP -56.755023   TA 0.880639 HA 0.834586
Update 306  TP -610.072203  HP -61.045889   TA 0.786654 HA 0.774436
Update 311  TP -612.131040  HP -61.428389   TA 0.784774 HA 0.774436
Update 316  TP -555.477929  HP -56.807726   TA 0.807331 HA 0.789474
Update 321  TP -393.037933  HP -47.349172   TA 0.863722 HA 0.834586
Update 326  TP -351.968273  HP -45.335780   TA 0.871241 HA 0.894737
Update 331  TP -337.473887  HP -46.508831   TA 0.875000 HA 0.894737
Update 336  TP -339.741938  HP -44.196772   TA 0.877820 HA 0.887218
Update 341  TP -336.342619  HP -44.211564   TA 0.876880 HA 0.887218
Update 346  TP -322.967512  HP -45.049632   TA 0.882519 HA 0.879699
Update 351  TP -323.897696  HP -47.748163   TA 0.887218 HA 0.887218
Update 356  TP -321.754975  HP -45.483744   TA 0.884398 HA 0.902256
Update 361  TP -323.528523  HP -46.937695   TA 0.888158 HA 0.879699
Update 366  TP -334.269882  HP -51.865085   TA 0.883459 HA 0.872180
Update 371  TP -308.817885  HP -43.953976   TA 0.897556 HA 0.887218
Update 376  TP -308.641745  HP -43.057911   TA 0.894737 HA 0.894737
Update 381  TP -293.370034  HP -41.637636   TA 0.900376 HA 0.894737
Update 386  TP -289.908528  HP -39.241285   TA 0.905075 HA 0.902256
Update 391  TP -286.407427  HP -39.324485   TA 0.906015 HA 0.887218
Update 396  TP -323.801733  HP -39.494747   TA 0.884398 HA 0.857143
Update 401  TP -1139.335807 HP -181.194653  TA 0.734962 HA 0.706767
Update 406  TP -845.217297  HP -145.291016  TA 0.788534 HA 0.736842
Update 411  TP -708.489915  HP -122.062183  TA 0.804511 HA 0.736842
Update 416  TP -589.225326  HP -100.913091  TA 0.828947 HA 0.774436
Update 421  TP -473.459119  HP -79.617395   TA 0.845865 HA 0.796992
Update 426  TP -456.520757  HP -75.193656   TA 0.853383 HA 0.812030
Update 431  TP -418.746995  HP -68.707612   TA 0.853383 HA 0.812030
Update 436  TP -453.687305  HP -72.867626   TA 0.855263 HA 0.842105
Update 441  TP -454.249040  HP -73.062660   TA 0.856203 HA 0.842105
Update 446  TP -446.993420  HP -71.187196   TA 0.856203 HA 0.842105
Update 451  TP -454.598036  HP -71.727706   TA 0.852444 HA 0.819549
Update 456  TP -512.142848  HP -82.988750   TA 0.851504 HA 0.796992
Update 461  TP -565.460551  HP -93.883870   TA 0.839286 HA 0.766917
Update 466  TP -521.144497  HP -86.798119   TA 0.850564 HA 0.781955
Update 471  TP -482.450294  HP -79.497508   TA 0.855263 HA 0.796992
Update 476  TP -435.000371  HP -70.247631   TA 0.862782 HA 0.827068
Update 481  TP -407.403748  HP -66.562197   TA 0.867481 HA 0.842105
Update 486  TP -416.735609  HP -68.393902   TA 0.866541 HA 0.834586
Update 491  TP -335.328723  HP -52.714022   TA 0.878759 HA 0.849624
Update 496  TP -407.686798  HP -67.872869   TA 0.872180 HA 0.774436
Update 501  TP -313.466591  HP -53.276068   TA 0.886278 HA 0.849624
Update 506  TP -311.258045  HP -53.598597   TA 0.886278 HA 0.849624
Update 511  TP -323.401879  HP -55.485871   TA 0.888158 HA 0.834586
Update 516  TP -348.637260  HP -59.918937   TA 0.883459 HA 0.842105
Update 521  TP -306.663457  HP -53.227269   TA 0.886278 HA 0.842105
Update 526  TP -312.758900  HP -48.909103   TA 0.874060 HA 0.819549
Update 531  TP -275.984106  HP -49.148778   TA 0.900376 HA 0.842105
Update 536  TP -276.652949  HP -51.009870   TA 0.904135 HA 0.849624
Update 541  TP -268.759423  HP -47.350756   TA 0.901316 HA 0.834586
Update 546  TP -269.330721  HP -47.386911   TA 0.902256 HA 0.834586
Update 551  TP -286.406119  HP -54.918118   TA 0.905075 HA 0.842105
Update 556  TP -285.817730  HP -54.953680   TA 0.906955 HA 0.842105
Update 561  TP -286.543784  HP -55.178743   TA 0.906955 HA 0.842105
Update 566  TP -277.241818  HP -54.251789   TA 0.908835 HA 0.842105
Update 571  TP -274.354150  HP -53.751747   TA 0.909774 HA 0.842105
Update 576  TP -277.979745  HP -54.797502   TA 0.908835 HA 0.842105
Update 581  TP -266.683671  HP -53.645483   TA 0.909774 HA 0.842105
Update 586  TP -248.820131  HP -46.772507   TA 0.913534 HA 0.842105
Update 591  TP -247.179859  HP -46.432969   TA 0.913534 HA 0.834586
Update 596  TP -243.959335  HP -46.540317   TA 0.914474 HA 0.842105
Update 601  TP -241.665623  HP -44.282439   TA 0.908835 HA 0.842105
Update 606  TP -241.827718  HP -44.299548   TA 0.908835 HA 0.842105
Update 611  TP -242.401802  HP -44.299976   TA 0.907895 HA 0.849624
Update 616  TP -225.689472  HP -43.377978   TA 0.927632 HA 0.857143
Update 621  TP -231.155267  HP -44.126177   TA 0.924812 HA 0.864662
Update 626  TP -229.139643  HP -44.076266   TA 0.925752 HA 0.864662
Update 631  TP -226.828828  HP -43.046147   TA 0.925752 HA 0.864662
Update 636  TP -225.437706  HP -42.699508   TA 0.925752 HA 0.864662
Update 641  TP -225.906460  HP -43.032122   TA 0.927632 HA 0.864662
Update 646  TP -223.162695  HP -42.559025   TA 0.924812 HA 0.872180
Update 651  TP -222.700216  HP -42.439224   TA 0.928571 HA 0.864662
Update 656  TP -222.722059  HP -42.583771   TA 0.928571 HA 0.864662
Update 661  TP -274.310571  HP -42.173163   TA 0.908835 HA 0.819549
Update 666  TP -358.480188  HP -53.101136   TA 0.866541 HA 0.789474
Update 671  TP -218.871854  HP -39.731280   TA 0.921053 HA 0.857143
Update 676  TP -209.704211  HP -37.760275   TA 0.921992 HA 0.879699
Update 681  TP -198.560400  HP -36.869748   TA 0.933271 HA 0.894737
Update 686  TP -197.826224  HP -36.707755   TA 0.933271 HA 0.894737
Update 691  TP -198.655141  HP -36.834976   TA 0.931391 HA 0.894737
Update 696  TP -199.935210  HP -36.862684   TA 0.931391 HA 0.887218
Update 701  TP -201.028859  HP -37.035274   TA 0.930451 HA 0.879699
Update 706  TP -200.897898  HP -37.131970   TA 0.931391 HA 0.879699
Update 711  TP -202.051354  HP -37.249490   TA 0.929511 HA 0.879699
Update 716  TP -200.656528  HP -37.273816   TA 0.930451 HA 0.879699
Update 721  TP -197.767014  HP -36.940982   TA 0.929511 HA 0.879699
Update 726  TP -211.564050  HP -38.013143   TA 0.926692 HA 0.864662
Update 731  TP -221.378201  HP -36.904729   TA 0.917293 HA 0.864662
Update 736  TP -253.103370  HP -39.507239   TA 0.901316 HA 0.834586
Update 741  TP -258.055031  HP -40.143243   TA 0.899436 HA 0.834586
Update 746  TP -186.948024  HP -39.229650   TA 0.934211 HA 0.872180
Update 751  TP -181.355020  HP -39.935733   TA 0.941729 HA 0.872180
Update 756  TP -183.072380  HP -39.562470   TA 0.933271 HA 0.872180
Update 761  TP -182.835189  HP -39.741767   TA 0.932331 HA 0.872180
Update 766  TP -175.189287  HP -41.256160   TA 0.941729 HA 0.887218
Update 771  TP -177.241215  HP -41.667541   TA 0.938910 HA 0.887218
Update 776  TP -174.784786  HP -41.236617   TA 0.944549 HA 0.887218
Update 781  TP -176.277735  HP -41.512734   TA 0.940789 HA 0.887218
Update 786  TP -179.732178  HP -42.471561   TA 0.939850 HA 0.894737
Update 791  TP -174.789376  HP -41.311321   TA 0.944549 HA 0.917293
Update 796  TP -174.211594  HP -41.112760   TA 0.944549 HA 0.917293
Update 801  TP -172.833457  HP -40.893924   TA 0.945489 HA 0.917293
Update 806  TP -179.769581  HP -44.065622   TA 0.939850 HA 0.902256
Update 811  TP -172.531644  HP -42.413535   TA 0.941729 HA 0.917293
Update 816  TP -173.985131  HP -42.070528   TA 0.943609 HA 0.917293
Update 821  TP -173.785749  HP -42.037905   TA 0.944549 HA 0.917293
Update 826  TP -187.497428  HP -44.028831   TA 0.936090 HA 0.902256
Update 831  TP -188.382297  HP -44.224869   TA 0.935150 HA 0.894737
Update 836  TP -177.826011  HP -42.042913   TA 0.941729 HA 0.902256
Update 841  TP -204.810022  HP -47.836478   TA 0.937030 HA 0.879699
Update 846  TP -202.177386  HP -47.196144   TA 0.936090 HA 0.879699
Update 851  TP -201.182148  HP -46.989697   TA 0.937030 HA 0.879699
Update 856  TP -165.377773  HP -41.115073   TA 0.951128 HA 0.894737
Update 861  TP -166.390375  HP -39.693556   TA 0.945489 HA 0.879699
Update 866  TP -160.771148  HP -39.273862   TA 0.953008 HA 0.894737
Update 871  TP -158.527804  HP -39.397144   TA 0.951128 HA 0.894737
Update 876  TP -159.606206  HP -39.630264   TA 0.951128 HA 0.894737
Update 881  TP -159.624323  HP -39.651510   TA 0.951128 HA 0.894737
Update 886  TP -159.350120  HP -39.600990   TA 0.952068 HA 0.894737
Update 891  TP -159.156198  HP -38.949375   TA 0.951128 HA 0.887218
Update 896  TP -152.787921  HP -37.358780   TA 0.953008 HA 0.894737
Update 901  TP -151.139076  HP -35.680051   TA 0.953947 HA 0.902256
Update 906  TP -150.251937  HP -35.433815   TA 0.954887 HA 0.909774
Update 911  TP -158.992120  HP -37.080554   TA 0.952068 HA 0.924812
Update 916  TP -152.206975  HP -35.730176   TA 0.953947 HA 0.917293
Update 921  TP -151.788943  HP -35.494941   TA 0.953008 HA 0.924812
Update 926  TP -151.221267  HP -35.358280   TA 0.953008 HA 0.917293
Update 931  TP -148.850016  HP -35.165356   TA 0.953008 HA 0.909774
Update 936  TP -137.953489  HP -35.802599   TA 0.957707 HA 0.909774
Update 941  TP -138.242068  HP -32.667226   TA 0.962406 HA 0.902256
Update 946  TP -136.489439  HP -32.393536   TA 0.961466 HA 0.909774
Update 951  TP -141.532896  HP -32.726237   TA 0.962406 HA 0.894737
Update 956  TP -141.006274  HP -32.751988   TA 0.962406 HA 0.894737
Update 961  TP -143.077572  HP -32.271514   TA 0.957707 HA 0.894737
Update 966  TP -209.525004  HP -37.212828   TA 0.916353 HA 0.857143
Update 971  TP -237.022734  HP -41.365749   TA 0.911654 HA 0.819549
Update 976  TP -236.456555  HP -41.318163   TA 0.911654 HA 0.819549
Update 981  TP -238.323369  HP -41.596260   TA 0.911654 HA 0.812030
Update 986  TP -190.088612  HP -35.730672   TA 0.932331 HA 0.857143
Update 991  TP -125.908772  HP -29.593233   TA 0.964286 HA 0.917293
Update 996  TP -125.944627  HP -29.246744   TA 0.966165 HA 0.917293
Update 1001 TP -125.990028  HP -29.270997   TA 0.966165 HA 0.917293
Update 1006 TP -125.828945  HP -29.308346   TA 0.966165 HA 0.917293
Update 1011 TP -126.581633  HP -29.662342   TA 0.963346 HA 0.917293
Update 1016 TP -116.508236  HP -30.082956   TA 0.967105 HA 0.924812
Update 1021 TP -114.268248  HP -31.212186   TA 0.969925 HA 0.924812
Update 1026 TP -114.366405  HP -31.040709   TA 0.969925 HA 0.924812
Update 1031 TP -114.343313  HP -31.085322   TA 0.969925 HA 0.924812
Update 1036 TP -114.387755  HP -30.830341   TA 0.969925 HA 0.924812
Update 1041 TP -115.060443  HP -30.781983   TA 0.970865 HA 0.924812
Update 1046 TP -114.288757  HP -31.039942   TA 0.969925 HA 0.924812
Update 1051 TP -115.585744  HP -30.766469   TA 0.967105 HA 0.917293
Update 1056 TP -114.505562  HP -30.630456   TA 0.969925 HA 0.924812
Update 1061 TP -114.114816  HP -30.655089   TA 0.969925 HA 0.924812
```

Hints
-

1.  As with the previous assignment, make sure that you debug on small
    datasets first (I've provided _toy text_ in the data directory to get you started).
1.  Certainly make sure that you do the unregularized version first
    and get it to work well.
1.  Use numpy functions whenever you can to make the computation faster.


