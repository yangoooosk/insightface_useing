# import insightface
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# import cv2
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
# from recognitions import recognition
#
#
# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
# cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头可以选择其他编号
# img=cv2.imread('./data/img1.jpg')
# face = app.get(img)
#
# while True:
#     ret, frame = cap.read()  # 读取一帧图像[{["face"]:233,[坐标]:233},{["face"]:233,[坐标]:233},{["face"]:233,[坐标]:233}]
#     faces = app.get(frame)
#     r = recognition(faces, face)
#     for i in r:
#         cv2.rectangle(frame,(int(i["bbox"][0]),int(i["bbox"][1])) ,(int(i["bbox"][2]),int(i["bbox"][3])) , (0, 255, 0), 3)
#         if i["face"]<1.3:
#             cv2.putText(frame, 'yangyuanyuan', (int(i["bbox"][0]),int(i["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
#     cv2.imshow('frame', frame)  # 显示图像
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
#         break


# import cv2
# import os
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
# import face_recognition # 导入face_recognition库
#
# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
#
# # 定义load_face_embeddings函数
# def load_face_embeddings(face_folder):
#     face_dict = {}
#     for file in os.listdir(face_folder):
#         name = os.path.splitext(file)[0]
#         img = face_recognition.load_image_file(os.path.join(face_folder, file))
#         embedding = face_recognition.face_encodings(img)[0]
#         face_dict[name] = embedding
#     return face_dict
#
# # 调用load_face_embeddings函数
# face_dict = load_face_embeddings('./data/faces')
#
# # 定义recognize_face函数
# def recognize_face(face_img):
#     face_embedding = face_recognition.face_encodings(face_img)[0]
#     best_name = None
#     best_score = float('inf')
#     for name, embedding in face_dict.items():
#         score = face_recognition.face_distance([embedding], face_embedding)[0]
#         if score < best_score:
#             best_name = name
#             best_score = score
#     return best_name, best_score
#
# try:
#     cap = cv2.VideoCapture(0)  # 0表示默认摄像头，如果有多个摄像头可以选择其他编号
#     while True:
#         ret, frame = cap.read()  # 读取一帧图像[{["face"]:233,[坐标]:233},{["face"]:233,[坐标]:233},{["face"]:233,[坐标]:233}]
#         if not ret:
#             break
#         faces = app.get(frame)
#         for i in faces:
#             cv2.rectangle(frame,(int(i["bbox"][0]),int(i["bbox"][1])) ,(int(i["bbox"][2]),int(i["bbox"][3])) , (0, 255, 0), 3)
#             # 调用recognize_face函数
#             name, score = recognize_face(frame[int(i["bbox"][1]):int(i["bbox"][3]), int(i["bbox"][0]):int(i["bbox"][2])])
#             # 根据相似度判断是否显示姓名
#             if score < 0.6:
#                 cv2.putText(frame, name, (int(i["bbox"][0]),int(i["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2, cv2.LINE_AA)
#             else:
#                 cv2.putText(frame, 'Unknown', (int(i["bbox"][0]),int(i["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2, cv2.LINE_AA)
#         cv2.imshow('frame', frame)  # 显示图像
#         if cv2.waitKey(10) & 0xFF == ord('q'):  # 按q键退出
#             break
# except Exception as e:
#     print(e)
# finally:
#     cap.release()
#     cv2.destroyAllWindows()
#
# import cv2
# import insightface
# from insightface.app import FaceAnalysis
# import os
# from insightface.data import get_image as ins_get_image
# import numpy as np
# from sklearn import preprocessing
#
# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
#
# # 定义load_face_embeddings函数
# def load_face_embeddings(face_folder):
#     face_dict = {}
#     for file in os.listdir(face_folder):
#         name = os.path.splitext(file)[0]
#         img = insightface.data.get_image(os.path.join(face_folder, file))
#         face = app.get(img)[0]
#         # embedding = np.array(face.embedding).reshape((1, -1))
#         # embedding = preprocessing.normalize(embedding)
#         face_dict[name] = face
#     return face_dict
#
# # 调用load_face_embeddings函数
# face_dict = load_face_embeddings('./data/faces')
# print(face_dict)
# # 定义feature_compare函数
# def feature_compare(feature1, feature2):
#     diff = np.subtract(feature1, feature2)
#     dist = np.sum(np.square(diff), 1)
#     return dist
#
# # 定义recognition函数
# def recognition(faces, face_dict, threshold=1.3):
#     r = list()
#     for face in faces:
#         # 开始人脸识别
#         rep = dict()
#         embedding = np.array(face.embedding).reshape((1, -1))
#         embedding = preprocessing.normalize(embedding)
#         best_name = None
#         best_score = float('inf')
#         for name, embedding1 in face_dict.items():
#             score = feature_compare(embedding, embedding1)
#             if score < best_score:
#                 best_name = name
#                 best_score = score
#         if best_score < threshold:
#             rep["name"] = best_name
#         else:
#             rep["name"] = 'Unknown'
#         rep["score"] = best_score
#         rep["bbox"] = face.bbox
#         r.append(rep)
#     return r
#
# try:
#     #cap = cv2.VideoCapture('D:/11/one.mp4')
#     cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0表示默认摄像头，如果有多个摄像头可以选择其他编号
#     while True:
#         ret, frame = cap.read()  # 读取一帧图像[{["face"]:233,[坐标]:233},{["face"]:233,[坐标]:233},{["face"]:233,[坐标]:233}]
#         if not ret:
#             break
#         faces = app.get(frame)
#         # 调用recognition函数
#         r = recognition(faces, face_dict[0], threshold=1.3)
#         for i in r:
#             cv2.rectangle(frame,(int(i["bbox"][0]),int(i["bbox"][1])) ,(int(i["bbox"][2]),int(i["bbox"][3])) , (0, 255, 0), 3)
#             # 根据返回的信息显示姓名和相似度分数
#             if i["name"] == 'Unknown':
#                 cv2.putText(frame, i["name"], (int(i["bbox"][0]),int(i["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2, cv2.LINE_AA)
#             else:
#                 cv2.putText(frame, i["name"] + ' (' + str(round(i["score"], 2)) + ')', (int(i["bbox"][0]),int(i["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2, cv2.LINE_AA)
#         cv2.imshow('frame', frame)  # 显示图像
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
#             break
# except Exception as e:
#     print(e)
# finally:
#      cap.release()
#      cv2.destroyAllWindows()

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
        face = app.get(img)[0]
        embedding = np.array(face.embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)
        face_dict[name] = embedding
    return face_dict


# 定义feature_compare函数
def feature_compare(feature1, feature2):
    diff = np.subtract(feature1, feature2)
    dist = np.sum(np.square(diff), 1)
    return dist


# 定义recognition函数
def recognition(faces, face_dict, threshold=1.3):
    r = list()
    for face in faces:
        rep = dict()
        embedding = np.array(face.embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)
        best_name = None
        best_score = float('inf')
        for name, embedding1 in face_dict.items():
            score = feature_compare(embedding, embedding1)
            if score < best_score:
                best_name = name
                best_score = score
        if best_score < threshold:
            rep["name"] = best_name
        else:
            rep["name"] = 'Unknown'
        rep["score"] = best_score
        rep["bbox"] = face.bbox
        r.append(rep)
    return r


# 调用load_face_embeddings函数
face_dict = load_face_embeddings('./data/faces')
print(face_dict)

try:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0表示默认摄像头
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        # 调用recognition函数
        r = recognition(faces, face_dict, threshold=1.3)
        for i in r:
            cv2.rectangle(frame, (int(i["bbox"][0]), int(i["bbox"][1])),
                          (int(i["bbox"][2]), int(i["bbox"][3])), (0, 255, 0), 3)
            if i["name"] == 'Unknown':
                cv2.putText(frame, i["name"], (int(i["bbox"][0]), int(i["bbox"][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, i["name"] + ' (' + str(round(i["score"][0], 2)) + ')', # 注意这里score是数组
                            (int(i["bbox"][0]), int(i["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2,
                            cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(e)
finally:
    cap.release()
    cv2.destroyAllWindows()
















