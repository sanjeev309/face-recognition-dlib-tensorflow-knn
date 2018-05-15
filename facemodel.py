import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
recogniser = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')
sp = dlib.shape_predictor('data/shape_predictor_5_face_landmarks.dat')


def getFaceEmbedding(img):
    dets = detector(img, 1)
    if len(dets)==0:
        print("No face detected!")

    else:
        if len(dets)==1:
            for k,d in enumerate(dets):
                # print(str(k) + " : "+ str(d))
                predictor = sp(img, d)
                descriptor = recogniser.compute_face_descriptor(img, predictor)
                return np.asarray(descriptor)
        else:
            descriptor = []
            for k,d in enumerate(dets):
                # print(str(k) + " : "+ str(d))
                predictor = sp(img, d)
                descriptor.append(recogniser.compute_face_descriptor(img, predictor))

            return np.asarray(np.squeeze(descriptor,axis=0))

def getNumFaces(img):

    dets = detector(img,1)
    if len(dets) == 0 :
        print("No faces detected in the image")

    else:
        print("Num of faces : "+ str(len(dets)))

    return len(dets)