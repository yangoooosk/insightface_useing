import insightface
from insightface.app import FaceAnalysis
import os
import cv2
import numpy as np
from sklearn import preprocessing

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


# 定义load_face_embeddings函数
# 遍历图片并提取人脸特征，将文件名作为键  人脸特征值存在face_dict中
def load_face_embeddings(face_folder):
    face_dict = {}
    for file in os.listdir(face_folder):# 获取文件夹中所有文件的文件名
        name = os.path.splitext(file)[0] # 提取文件名用作人名
        img = cv2.imread(os.path.join(face_folder, file))
        face = app.get(img)
        # embedding = np.array(face.embedding).reshape((1, -1))
        # embedding = preprocessing.normalize(embedding)
        face_dict[name] = face
    return face_dict


# 调用load_face_embeddings函数  加载文件夹中的人脸嵌入向量
face_dict = load_face_embeddings('./data/faces')


# 定义feature_compare函数  计算两个嵌入向量之间的欧几里得距离
def feature_compare(feature1, feature2):
    diff = np.subtract(feature1, feature2)
    dist = np.sum(np.square(diff), 1)
    return dist


# 定义recognition函数
def recognition(faces, face_p):
    r = list()
    embedding1 = np.array(face_p[0].embedding).reshape((1, -1))
    embedding1 = preprocessing.normalize(embedding1)
    for face in faces:
        # 开始人脸识别
        rep = dict()
        embedding = np.array(face.embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)
        rep["face"] = feature_compare(embedding, embedding1)
        rep["bbox"] = face.bbox
        rep["age"] = face.age
        gender = 'man'
        if face.gender == 0:
            gender = 'woman'
        rep["gender"] = gender
        r.append(rep)
    return r


# 读取图片文件夹中的所有图片并进行处理
image_folder = './data/images'  # 输入图像文件夹路径
output_folder = './data/output'  # 输出图像文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)
    faces = app.get(frame)
    for k, v in face_dict.items():
        r = recognition(faces, v)
        for i in r:
            cv2.rectangle(frame, (int(i["bbox"][0]), int(i["bbox"][1])), (int(i["bbox"][2]), int(i["bbox"][3])),
                          (0, 255, 0), 3)
            cv2.putText(frame, str(len(r)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            cv2.putText(frame, i["gender"], (int(i["bbox"][0]), int(i["bbox"][1]) + 40 * 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (55, 255, 155), 2)
            cv2.putText(frame, str(i["age"]), (int(i["bbox"][0]), int(i["bbox"][1]) + 40 * 3), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (55, 255, 155), 2)
            if i["face"] < 1.3:
                cv2.putText(frame, k, (int(i["bbox"][0]), int(i["bbox"][1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (55, 255, 155), 2)
                cv2.putText(frame, str(i["face"]), (int(i["bbox"][0]), int(i["bbox"][1]) + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)

    # 保存处理过的图像
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, frame)

    # 显示图像（可选）
    cv2.imshow('frame', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):  # 按q键退出
        break

cv2.destroyAllWindows()
