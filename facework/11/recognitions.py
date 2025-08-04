import insightface  # 导入InsightFace库，用于进行人脸检测和识别任务
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘制图像和图形

import cv2  # 导入OpenCV库，用于处理图像和视频
import numpy as np  # 导入NumPy库，用于执行数组操作和数学计算
from insightface.app import FaceAnalysis  # 从InsightFace库中导入FaceAnalysis类，用于人脸分析
from insightface.data import get_image as ins_get_image  # 导入InsightFace的内置图像加载函数，用于获取示例图像
from sklearn import preprocessing  # 导入sklearn库，用于数据预处理（例如，标准化和归一化）

# 定义recognition函数，接受检测到的人脸信息和待比对的人脸数据，进行人脸识别
def recognition(faces, face_p):
    r = list()  # 初始化一个空列表，用于存储识别结果
    # 获取待比对的人脸特征embedding，并进行标准化处理
    embedding1 = np.array(face_p[0].embedding).reshape((1, -1))  # 将face_p的embedding转为一维数组
    embedding1 = preprocessing.normalize(embedding1)  # 对embedding进行归一化处理

    for face in faces:
        # 遍历每个检测到的人脸
        rep = dict()  # 创建一个字典来存储当前人脸的相关信息
        # 获取当前人脸的embedding，并进行标准化处理
        embedding = np.array(face.embedding).reshape((1, -1))  # 将当前人脸的embedding转为一维数组
        embedding = preprocessing.normalize(embedding)  # 对embedding进行归一化处理

        # 计算当前人脸与待比对人脸之间的相似度（通过欧几里得距离）
        rep["face"] = feature_compare(embedding, embedding1)  # 存储计算出的相似度
        rep["bbox"] = face.bbox  # 存储当前人脸的边界框（bbox），格式为[x_min, y_min, x_max, y_max]

        r.append(rep)  # 将当前人脸的识别信息添加到结果列表中

    return r  # 返回所有识别到的人脸信息，包括相似度和边界框

# 定义feature_compare函数，计算两个特征向量之间的欧几里得距离
def feature_compare(feature1, feature2):
    diff = np.subtract(feature1, feature2)  # 计算两个特征向量的差值
    dist = np.sum(np.square(diff), 1)  # 计算差值的平方和，即欧几里得距离
    return dist  # 返回计算出的欧几里得距离
