import cv2  # 导入OpenCV库，用于图像和视频流处理
import insightface  # 导入InsightFace库，用于人脸检测和识别
from insightface.app import FaceAnalysis  # 导入FaceAnalysis类，用于进行人脸分析
import os  # 导入os库，用于文件和目录操作
from insightface.data import get_image as ins_get_image  # 导入图片加载函数
import numpy as np  # 导入NumPy库，用于数值计算和数组处理
from sklearn import preprocessing  # 导入sklearn的预处理模块，用于归一化和标准化处理

# 初始化FaceAnalysis应用，指定使用CUDA和CPU作为执行提供者
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# 准备人脸检测模型，ctx_id=0表示使用第0个GPU，det_size指定图像输入尺寸
app.prepare(ctx_id=0, det_size=(640, 640))

# 定义load_face_embeddings函数，加载并归一化人脸特征
def load_face_embeddings(face_folder):
    face_dict = {}  # 初始化一个空字典，用于存储人脸名称和对应的特征
    for file in os.listdir(face_folder):  # 遍历指定文件夹中的所有文件
        name = os.path.splitext(file)[0]  # 获取文件名（不包含扩展名），作为人名
        # 加载图像文件并提取其中的人脸特征
        img = insightface.data.get_image(os.path.join(face_folder, file))
        face = app.get(img)[0]  # 获取图像中的人脸（返回一个人脸列表，这里取第一个）
        embedding = np.array(face.embedding).reshape((1, -1))  # 提取人脸特征并转化为一维数组
        embedding = preprocessing.normalize(embedding)  # 对特征进行归一化处理
        face_dict[name] = embedding  # 将人名和特征存入字典
    return face_dict  # 返回包含所有已知人脸特征的字典

# 定义feature_compare函数，计算两个特征向量之间的相似度
def feature_compare(feature1, feature2):
    diff = np.subtract(feature1, feature2)  # 计算两个特征向量的差值
    dist = np.sum(np.square(diff), 1)  # 计算差值的平方和，得到欧几里得距离
    return dist  # 返回计算得到的距离值

# 定义recognition函数，进行人脸识别
def recognition(faces, face_dict, threshold=1.3):
    r = list()  # 初始化一个空列表，用于存储识别结果
    for face in faces:  # 遍历检测到的所有人脸
        rep = dict()  # 初始化一个字典，用于存储当前人脸的识别信息
        embedding = np.array(face.embedding).reshape((1, -1))  # 提取并转换当前人脸的特征
        embedding = preprocessing.normalize(embedding)  # 对当前人脸特征进行归一化处理
        best_name = None  # 初始化最匹配的人名
        best_score = float('inf')  # 初始化最小匹配得分为无穷大
        # 遍历已知人脸字典，计算当前人脸与每个已知人脸的匹配得分
        for name, embedding1 in face_dict.items():
            score = feature_compare(embedding, embedding1)  # 计算当前人脸与已知人脸的距离
            if score < best_score:  # 如果当前得分更小，说明更匹配
                best_name = name  # 更新最佳匹配的人名
                best_score = score  # 更新最佳匹配得分
        # 判断匹配得分是否小于设定的阈值，如果小于阈值则认为识别成功
        if best_score < threshold:
            rep["name"] = best_name  # 记录识别到的姓名
        else:
            rep["name"] = 'Unknown'  # 否则标记为"Unknown"（未知）
        rep["score"] = best_score  # 记录匹配得分
        rep["bbox"] = face.bbox  # 记录人脸的位置框（bounding box）
        r.append(rep)  # 将识别结果添加到返回结果列表
    return r  # 返回所有识别到的结果

# 调用load_face_embeddings函数加载已知人脸特征，并将结果保存在face_dict字典中
face_dict = load_face_embeddings('./data/faces')
print(face_dict)  # 打印加载的人脸特征字典

# 开始视频捕捉并处理每一帧
try:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 使用默认摄像头进行视频捕捉
    while True:
        ret, frame = cap.read()  # 读取一帧视频
        if not ret:
            break  # 如果读取失败，跳出循环
        faces = app.get(frame)  # 使用InsightFace检测视频帧中的人脸
        # 调用recognition函数进行人脸识别
        r = recognition(faces, face_dict, threshold=1.3)
        for i in r:  # 遍历识别结果
            # 在图像中绘制人脸框
            cv2.rectangle(frame, (int(i["bbox"][0]), int(i["bbox"][1])),
                          (int(i["bbox"][2]), int(i["bbox"][3])), (0, 255, 0), 3)
            # 如果人脸识别结果是"Unknown"，显示"Unknown"
            if i["name"] == 'Unknown':
                cv2.putText(frame, i["name"], (int(i["bbox"][0]), int(i["bbox"][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2, cv2.LINE_AA)
            else:
                # 如果识别到人名，显示人名和相似度得分
                cv2.putText(frame, i["name"] + ' (' + str(round(i["score"][0], 2)) + ')',
                            (int(i["bbox"][0]), int(i["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2,
                            cv2.LINE_AA)
        # 显示处理后的帧图像
        cv2.imshow('frame', frame)
        # 如果按下‘q’键，则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:  # 捕获并打印异常
    print(e)
finally:
    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
