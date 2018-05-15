import numpy as np
import util
import facemodel
from skimage import io


"""FSM to manage different functions of the system
    Give approriate input when prompted 
"""

knnData = []


if __name__ == "__main__":

    images = util.getX()
    labels = util.getY()

    # num = util.getMaxPeople()

    for image,label in zip(images,labels):
        print(str(image) + " : "+ str(label))

        readImage = io.imread('Dataset/'+ image)
        vector = facemodel.getFaceEmbedding(readImage)
        knnData.append([vector,label])


    np.vstack(knnData)
    np.save('data/knnData',knnData)