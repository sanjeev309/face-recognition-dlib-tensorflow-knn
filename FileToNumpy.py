
import numpy as np


x = []
y = []
Y = []

with open('Dataset/faceLabels.txt') as file:
        for line in file:
            a,b = line.split(',')
            x.append(a)
            y.append(b)

for element in y:
    Y.append(int(element.replace('\n','')))


np.save('data/Ylabels',Y)
np.save('data/Ximages',x)