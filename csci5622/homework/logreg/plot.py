import matplotlib.pyplot as plt
import numpy as np


y = np.array([.1, .01 , .001, .0001, .00001, .000001, .0000001, .00000001, 
              .000000001, .0000000001, .00000000001])
x = np.array([.954887, .954887, .947368, .932331, .902256, .879699, .879699, 
              .879699, .879699, .879699, .879699])
plt.yscale('log', linthresy=.1)
plt.ylim(.000000000001, ymax=1)
plt.xlim(xmin=.87, xmax=.96)
plt.plot(x,y, 'ro')
plt.xlabel('Test Accuracy at Three Passes')
plt.ylabel('eta (learning rate)')
plt.show()
