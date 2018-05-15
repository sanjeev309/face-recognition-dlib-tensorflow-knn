"""This is a utility file for basic IO operations"""

import numpy as np

def getX():
    return np.load('data/Ximages.npy')

def getY():
    return np.array(np.load('data/Ylabels.npy'),dtype=np.int8)

def getMaxPeople():
    y = np.load('data/Ylabels.npy')
    return y[-1] + 1

def getKNNx():
    return np.load('data/knnData.npy')[:,0]