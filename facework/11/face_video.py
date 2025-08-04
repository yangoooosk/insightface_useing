import cv2  # 导入OpenCV库，用于图像和视频处理
import insightface  # 导入InsightFace库，用于人脸检测与识别
from insightface.app import FaceAnalysis  # 从InsightFace中导入FaceAnalysis类，用于人脸分析
import os  # 导入os库，用于文件和目录操作
from insightface.data import get_image as ins_get_image  # 从InsightFace中导入get_image函数，获取图像数据
import numpy as np  # 导入NumPy库，用于数组操作
from sklearn import preprocessing  # 导入sklearn的preprocessing模块，用于特征预处理

# 初始化FaceAnalysis应用，指定使用CUDA加速（如果有GPU）和CPU备选方案
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# 准备人脸检测和分析模型，ctx_id=0表示使用默认的GPU（如果可用），det_size为输入图像的大小
app.prepare(ctx_id=0, det_size=(640, 640))


# 定义一个函数，用于加载人脸特征（embeddings）到字典中
def load_face_embeddings(face_folder):
    """
    加载指定文件夹中的所有人脸图像，提取人脸特征，并将其存储在字典中。

    :param face_folder: 存放人脸图像的文件夹路径
    :return: 包含人脸特征的字典，键为图像名称，值为提取的特征
    """
    face_dict = {}  # 用于存储人脸特征的字典
    for file in os.listdir(face_folder):  # 遍历文件夹中的所有文件
        # 提取文件名作为人名
        name = os.path.splitext(file)[0]
        # 读取图像数据
        img = insightface.data.get_image(os.path.join(face_folder, file))
        # 使用InsightFace获取图像中的人脸信息
        face = app.get(img)
        # 将人脸特征存入字典
        face_dict[name] = face
    return face_dict


# 调用load_face_embeddings函数，加载指定文件夹中的人脸特征数据
face_dict = load_face_embeddings('./data/faces')


# 定义一个函数，用于计算两个特征向量之间的欧式距离
def feature_compare(feature1, feature2):
    """
    比较两个特征向量的差异，返回它们之间的欧式距离。

    :param feature1: 第一个特征向量
    :param feature2: 第二个特征向量
    :return: 两个特征向量之间的欧式距离
    """
    diff = np.subtract(feature1, feature2)  # 计算特征向量的差异
    dist = np.sum(np.square(diff), 1)  # 计算差异的平方和，即欧式距离的平方
    return dist


# 定义人脸识别函数，用于识别视频流中的人脸并与已知人脸库进行匹配
def recognition(faces, face_p):
    """
    进行人脸识别，比较每一张检测到的人脸与指定的目标人脸。

    :param faces: 当前视频帧中的所有人脸
    :param face_p: 已知人脸特征
    :return: 包含识别结果的列表，每个元素包括人脸的匹配信息、年龄、性别等
    """
    r = list()  # 用于存储识别结果的列表
    # 获取目标人脸的特征并进行归一化处理
    embedding1 = np.array(face_p[0].embedding).reshape((1, -1))
    embedding1 = preprocessing.normalize(embedding1)

    for face in faces:
        # 对每一张检测到的人脸进行识别
        rep = dict()  # 存储每个人脸的识别结果
        embedding = np.array(face.embedding).reshape((1, -1))  # 获取人脸特征并调整形状
        embedding = preprocessing.normalize(embedding)  # 对人脸特征进行归一化处理

        # 计算当前人脸与目标人脸的匹配度（欧式距离）
        rep["face"] = feature_compare(embedding, embedding1)
        rep["bbox"] = face.bbox  # 获取人脸的位置（bounding box）
        rep["age"] = face.age  # 获取人脸的年龄
        # 判断人脸性别（0表示女性，1表示男性）
        gender = 'man' if face.gender == 1 else 'woman'
        rep["gender"] = gender  # 存储性别

        r.append(rep)  # 将识别结果添加到列表中
    return r


# 调用load_face_embeddings函数，加载指定文件夹中的人脸特征数据
face_dict = load_face_embeddings('./data/faces')

# 打开摄像头，准备捕获视频流
cap = cv2.VideoCapture(0)
# video_path = 'D:/11/one.mp4'  # 替换为您视频文件的路径
# cap = cv2.VideoCapture(video_path)  # 如果要读取视频文件，可以替换为视频路径

# 开始循环处理每一帧视频
while True:
    ret, frame = cap.read()  # 读取一帧图像
    # 使用人脸分析模型获取当前帧中的所有人脸信息

