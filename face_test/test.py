# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
#
# app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
#
# img = cv2.imread("./data/faces/chengyi.png")
#
# if img is None:
#     print("错误：无法加载图片。请检查图片路径。")
# else:
#     faces = app.get(img)
#
#     # 手动绘制边界框和关键点
#     for face in faces:
#         bbox = face.bbox.astype(np.int32)
#         landmarks = face.landmark_2d_106.astype(np.int32)  # RetinaFace 使用 face.landmark
#
#         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#         for landmark in landmarks:
#             cv2.circle(img, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)
#
#     cv2.imshow("img", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # cv2.imwrite("multi_people_output.jpg", img)  # 保存图片 (可选)


import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# 创建输出文件夹
output_folder = "test_out"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

test_folder = "./data/test"  #  测试图片文件夹路径

for filename in os.listdir(test_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # 检查文件扩展名
        img_path = os.path.join(test_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"错误：无法加载图片: {filename}。请检查图片路径。")
            continue  # 跳过无法加载的图片

        faces = app.get(img)

        # 绘制边界框和关键点
        for face in faces:
            bbox = face.bbox.astype(np.int32)
            landmarks = face.landmark_2d_106.astype(np.int32)

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            for landmark in landmarks:
                cv2.circle(img, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)

        # 构建输出文件名
        name, ext = os.path.splitext(filename)  # 分离文件名和扩展名
        output_filename = name + "_out" + ext
        output_path = os.path.join(output_folder, output_filename)

        cv2.imwrite(output_path, img)  # 保存图片




