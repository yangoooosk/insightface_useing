import cv2  # 导入OpenCV库，用于图像和视频处理
import numpy as np  # 导入NumPy库，用于数组操作
import os  # 导入os库，用于文件和目录操作
from insightface.app import FaceAnalysis  # 导入InsightFace库中的FaceAnalysis类，用于人脸分析

# 创建输出文件夹，如果该文件夹不存在，则创建它
output_folder = "test_out"
if not os.path.exists(output_folder):  # 检查输出文件夹是否存在
    os.makedirs(output_folder)  # 如果不存在，则创建该文件夹

# 初始化FaceAnalysis应用，指定使用CUDA加速（如果有GPU）和CPU备选方案
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# 准备人脸检测和分析模型，ctx_id=0表示使用默认的GPU（如果可用），det_size为输入图像的大小
app.prepare(ctx_id=0, det_size=(640, 640))

# 设置存放测试图片的文件夹路径
test_folder = "./data/test"  # 测试图片所在的文件夹路径

# 遍历指定文件夹中的所有文件
for filename in os.listdir(test_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # 检查文件扩展名，处理图片文件
        img_path = os.path.join(test_folder, filename)  # 获取图片的完整路径
        img = cv2.imread(img_path)  # 使用OpenCV读取图片

        if img is None:  # 如果图片加载失败，输出错误并跳过该文件
            print(f"错误：无法加载图片: {filename}。请检查图片路径。")
            continue  # 跳过当前无法加载的图片，继续下一个图片

        # 使用FaceAnalysis模型进行人脸检测，返回检测到的人脸信息
        faces = app.get(img)

        # 遍历检测到的每个人脸并绘制边界框和关键点
        for face in faces:
            bbox = face.bbox.astype(np.int32)  # 获取人脸的边界框，并转换为整型
            landmarks = face.landmark_2d_106.astype(np.int32)  # 获取人脸的106个关键点，并转换为整型

            # 在图像上绘制边界框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # 绘制绿色边框

            # 在图像上绘制每个关键点
            for landmark in landmarks:
                cv2.circle(img, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)  # 绘制红色圆点，表示关键点位置

        # 构建输出文件名，使用原文件名加上 "_out" 后缀
        name, ext = os.path.splitext(filename)  # 分离文件名和扩展名
        output_filename = name + "_out" + ext  # 创建新的输出文件名，添加"_out"后缀
        output_path = os.path.join(output_folder, output_filename)  # 构建输出文件的完整路径

        # 保存处理后的图片到输出文件夹
        cv2.imwrite(output_path, img)  # 使用OpenCV将处理后的图片保存到指定路径
