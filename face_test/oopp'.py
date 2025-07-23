import cv2
import insightface
from insightface.app import FaceAnalysis
import os
from insightface.data import get_image as ins_get_image
import numpy as np
from sklearn import preprocessing

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 定义load_face_embeddings函数
def load_face_embeddings(face_folder):
    face_dict = {}
    for file in os.listdir(face_folder):
        name = os.path.splitext(file)[0]
        img = insightface.data.get_image(os.path.join(face_folder, file))
        face = app.get(img)
        # embedding = np.array(face.embedding).reshape((1, -1))
        # embedding = preprocessing.normalize(embedding)
        face_dict[name] = face
    return face_dict

# 调用load_face_embeddings函数
face_dict = load_face_embeddings('./data/faces')
# 定义feature_compare函数
def feature_compare(feature1, feature2):
    diff = np.subtract(feature1, feature2)
    dist = np.sum(np.square(diff), 1)
    return dist

# 定义recognition函数
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
        rep["age"] = face.age
        gender = 'man'
        if face.gender == 0:
            gender = 'woman'
        rep["gender"] = gender
        r.append(rep)
    return r


face_dict = load_face_embeddings('./data/faces')


cap = cv2.VideoCapture(0)
#video_path = 'D:/11/one.mp4'  # 替换为您视频文件的路径
#cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()  # 读取一帧图像[{["face"]:233,[坐标]:233},{["face"]:233,[坐标]:233},{["face"]:233,[坐标]:233}]
    faces = app.get(frame)
    for k, v in face_dict.items():
        r = recognition(faces, v)
        for i in r:
            cv2.rectangle(frame,(int(i["bbox"][0]),int(i["bbox"][1])) ,(int(i["bbox"][2]),int(i["bbox"][3])) , (0, 255, 0), 3)
            cv2.putText(frame, str(len(r)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            cv2.putText(frame, i["gender"], (int(i["bbox"][0]), int(i["bbox"][1])+40*2), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            cv2.putText(frame, str(i["age"]), (int(i["bbox"][0]), int(i["bbox"][1]) + 40*3), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            if i["face"]<1.3:
                cv2.putText(frame, k, (int(i["bbox"][0]),int(i["bbox"][1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
                cv2.putText(frame, str(i["face"]), (int(i["bbox"][0]), int(i["bbox"][1])+40), cv2.FONT_HERSHEY_SIMPLEX, 1,(55, 255, 155), 2)
    cv2.imshow('frame', frame)  # 显示图像
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break





