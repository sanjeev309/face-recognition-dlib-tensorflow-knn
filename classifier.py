import knnModel as inference_model
import facemodel as initial_model
from skimage import io

import numpy as np
"""This is the inference module which detects and recognises faces in a given image
    Kindly change the path of the image in the below mentioned method
"""


test = io.imread("testImage.jpg")

numFace = initial_model.getNumFaces(test)

if numFace==1:

    descriptor = initial_model.getFaceEmbedding(test)
    inference_model.KnnModel(descriptor,infer = True)

else:
    for face in range(numFace):
        print('Face : ' + str(face + 1))
        descriptor = initial_model.getFaceEmbedding(test)
        inference_model.KnnModel(descriptor[face], infer=True)
