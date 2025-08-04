import insightface  # 导入InsightFace库，用于人脸检测和特征提取
from insightface.app import FaceAnalysis  # 从InsightFace库中导入FaceAnalysis类，负责人脸分析
import os  # 导入os库，用于文件和目录操作
import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数组操作
from sklearn import preprocessing  # 导入sklearn库，用于数据标准化（如归一化）

# 初始化FaceAnalysis应用，指定使用CUDA加速（如果有GPU）和CPU备选方案
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# 准备人脸检测和分析模型，ctx_id=0表示使用默认的GPU（如果可用），det_size为输入图像的大小
app.prepare(ctx_id=0, det_size=(640, 640))

# 定义load_face_embeddings函数
# 该函数用于遍历指定文件夹中的图片并提取人脸特征，返回一个字典，其中包含人脸特征（face embeddings）
def load_face_embeddings(face_folder):
    face_dict = {}  # 创建一个字典，用于存储每个文件的特征
    for file in os.listdir(face_folder):  # 遍历文件夹中的所有文件
        name = os.path.splitext(file)[0]  # 获取文件名（不包括扩展名）作为人的名字
        img = cv2.imread(os.path.join(face_folder, file))  # 读取图片文件
        face = app.get(img)  # 使用FaceAnalysis模型进行人脸检测和特征提取
        # embedding = np.array(face.embedding).reshape((1, -1))  # 提取人脸特征（嵌入向量）
        # embedding = preprocessing.normalize(embedding)  # 对嵌入向量进行归一化
        face_dict[name] = face  # 将提取的人脸特征存入字典，键为文件名（人的名字）
    return face_dict  # 返回字典，包含每个人的人脸特征

# 调用load_face_embeddings函数，加载指定文件夹中的所有人脸嵌入向量
face_dict = load_face_embeddings('./data/faces')

# 定义feature_compare函数
# 该函数用于计算两个嵌入向量（feature1和feature2）之间的欧几里得距离，返回距离值
def feature_compare(feature1, feature2):
    diff = np.subtract(feature1, feature2)  # 计算两个嵌入向量的差异
    dist = np.sum(np.square(diff), 1)  # 计算差异的平方和，得到欧几里得距离
    return dist  # 返回计算得到的距离

# 定义recognition函数
# 该函数用于对比检测到的人脸特征与输入的参考人脸特征，返回识别结果
def recognition(faces, face_p):
    r = list()  # 用于存储识别结果的列表
    embedding1 = np.array(face_p[0].embedding).reshape((1, -1))  # 提取输入人脸的嵌入向量
    embedding1 = preprocessing.normalize(embedding1)  # 对嵌入向量进行归一化处理
    for face in faces:  # 遍历检测到的每一张人脸
        rep = dict()  # 用于存储当前人脸的识别结果
        embedding = np.array(face.embedding).reshape((1, -1))  # 提取当前人脸的嵌入向量
        embedding = preprocessing.normalize(embedding)  # 对嵌入向量进行归一化处理
        rep["face"] = feature_compare(embedding, embedding1)  # 计算当前人脸与输入人脸的特征差异（距离）
        rep["bbox"] = face.bbox  # 存储人脸的边界框
        rep["age"] = face.age  # 存储人脸的年龄
        gender = 'man'  # 默认性别为男性
        if face.gender == 0:  # 如果性别为女性
            gender = 'woman'
        rep["gender"] = gender  # 存储性别信息
        r.append(rep)  # 将当前人脸的识别结果添加到结果列表
    return r  # 返回识别结果列表

# 读取图片文件夹中的所有图片并进行处理
image_folder = './data/images'  # 输入图像文件夹路径
output_folder = './data/output'  # 输出图像文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):  # 如果输出文件夹不存在
    os.makedirs(output_folder)  # 创建输出文件夹

# 遍历输入图像文件夹中的所有图片文件
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)  # 获取图片的完整路径
    frame = cv2.imread(image_path)  # 读取图片

    faces = app.get(frame)  # 使用FaceAnalysis模型进行人脸检测

    # 遍历加载的face_dict字典，逐个比对检测到的人脸和face_dict中的每个人
    for k, v in face_dict.items():  # k是人名，v是该人脸特征
        r = recognition(faces, v)  # 进行人脸识别，得到识别结果
        # 遍历识别结果并在图像上绘制相关信息
        for i in r:
            # 在图像上绘制人脸边界框
            cv2.rectangle(frame, (int(i["bbox"][0]), int(i["bbox"][1])),
                          (int(i["bbox"][2]), int(i["bbox"][3])), (0, 255, 0), 3)
            # 在图像上绘制识别到的人数
            cv2.putText(frame, str(len(r)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            # 绘制性别信息
            cv2.putText(frame, i["gender"], (int(i["bbox"][0]), int(i["bbox"][1]) + 40 * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            # 绘制年龄信息
            cv2.putText(frame, str(i["age"]), (int(i["bbox"][0]), int(i["bbox"][1]) + 40 * 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            # 如果人脸匹配度小于1.3，则显示该人的名字和匹配度
            if i["face"] < 1.3:
                cv2.putText(frame, k, (int(i["bbox"][0]), int(i["bbox"][1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
                cv2.putText(frame, str(i["face"]), (int(i["bbox"][0]), int(i["bbox"][1]) + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    # 保存处理过的图像到输出文件夹
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, frame)

    # 显示图像（可选）
    cv2.imshow('frame', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):  # 按下q键退出
        break

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
