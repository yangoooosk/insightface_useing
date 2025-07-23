import insightface
import matplotlib.pyplot as plt

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from sklearn import preprocessing

def recognition(faces, face_p):
    r=list()
    embedding1 = np.array(face_p[0].embedding).reshape((1, -1))
    embedding1 = preprocessing.normalize(embedding1)
    for face in faces:
        # 开始人脸识别
        rep=dict()
        embedding = np.array(face.embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)
        rep["face"]=feature_compare(embedding,embedding1)
        rep["bbox"]=face.bbox
        r.append(rep)
    return r


def feature_compare(feature1, feature2):
    diff = np.subtract(feature1, feature2)
    dist = np.sum(np.square(diff), 1)
    return dist



