import os  # 导入os模块，用于文件和目录操作

import cv2  # 导入OpenCV库，用于图像处理
import insightface  # 导入InsightFace库，用于人脸检测和识别
import numpy as np  # 导入NumPy库，用于数组操作和数值计算
from sklearn import preprocessing  # 导入sklearn中的预处理模块，用于特征标准化和归一化


# 定义FaceRecognition类，封装了人脸识别的相关功能
class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
        """
        初始化人脸识别系统的参数和模型。

        :param gpu_id: 使用的GPU设备编号，默认是0表示使用第一个GPU（如果有多个GPU时）。
        :param face_db: 存放已知人脸数据的文件夹路径，默认值为'face_db'。
        :param threshold: 识别的阈值，表示人脸匹配的最小相似度，默认值为1.24。
        :param det_thresh: 人脸检测的阈值，默认值为0.50，用于控制人脸检测的灵敏度。
        :param det_size: 输入图像的尺寸，默认值为(640, 640)，用于调整输入图像大小以便更好地进行检测。
        """
        self.gpu_id = gpu_id  # GPU设备编号
        self.face_db = face_db  # 人脸数据库路径
        self.threshold = threshold  # 人脸识别阈值
        self.det_thresh = det_thresh  # 人脸检测阈值
        self.det_size = det_size  # 人脸检测时的输入尺寸

        # 初始化InsightFace的人脸识别模型。'allowed_modules'限制为None，表示加载所有模块。
        # 'providers'指定CUDAExecutionProvider，表示使用GPU加速（如果可用）。
        self.model = insightface.app.FaceAnalysis(
            root='./',  # 模型根目录
            allowed_modules=None,  # 加载所有模块（如果需要，可以选择加载特定模块）
            providers=['CUDAExecutionProvider']  # 使用CUDA加速（如果GPU可用）
        )
        # 准备人脸检测模型，传入GPU设备编号、检测阈值、检测图像大小等参数。
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)

        # 初始化空列表，用于存储已加载的人脸特征
        self.faces_embedding = list()

        # 调用load_faces函数加载人脸数据库中的所有人脸数据
        self.load_faces(self.face_db)

    # 加载人脸数据库中的所有人脸及其特征
    def load_faces(self, face_db_path):
        """
        加载指定路径中的所有人脸图像并提取其特征。

        :param face_db_path: 存放已知人脸图像的文件夹路径。
        """
        # 如果人脸数据库路径不存在，则创建该目录
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)

        # 遍历数据库文件夹，获取所有子目录、文件及文件夹中的文件
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                # 使用OpenCV读取图像，确保支持中文路径（使用np.fromfile读取）
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)

                # 从文件名中提取用户名称（假设文件名为“姓名.jpg”，文件名去掉扩展名即为用户名称）
                user_name = file.split(".")[0]

                # 使用人脸识别模型提取图像中的人脸特征
                face = self.model.get(input_image)[0]

                # 获取该人脸的嵌入特征（embedding），并调整其形状以适应后续处理
                embedding = np.array(face.embedding).reshape((1, -1))

                # 对人脸特征进行归一化处理（确保特征向量在统一的尺度上）
                embedding = preprocessing.normalize(embedding)

                # 将提取到的人脸特征（包括用户名和特征向量）添加到faces_embedding列表中
                self.faces_embedding.append({
                    "user_name": user_name,  # 存储用户名称
                    "feature": embedding  # 存储该用户的人脸特征
                })
