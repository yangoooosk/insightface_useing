

# InsightFace 教程

`InsightFace` 是一个功能强大的 Python 库，专为 2D 和 3D 人脸分析而设计。它集成了人脸检测、人脸识别和人脸对齐等多种先进的算法和模型，是开发人脸相关应用的利器。

## 安装

### 使用 pip 安装

最简单的安装方式是通过 `pip` 命令：

```bash
pip install -U insightface
```

### 从 GitHub 源码安装

也可以直接从 GitHub 克隆源码进行安装，这种方式可以获取到最新的开发版本：

```bash
git clone https://github.com/deepinsight/insightface
cd insightface/python-package
pip install -e .
```

## 核心组件：FaceAnalysis

在 `InsightFace` 中，核心的功能都集成在 `FaceAnalysis` 类中。我们需要先创建一个该类的实例，并加载所需的模型。

```python
import insightface
from insightface.app import FaceAnalysis

# 初始化 FaceAnalysis，建议使用 GPU 以获得更好的性能
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# 准备模型，ctx_id=0 表示使用第一个 GPU，det_size 用于指定检测模型的输入尺寸
app.prepare(ctx_id=0, det_size=(640, 640))
```

## 1. 人脸检测

人脸检测是从图片中找出所有人脸的位置。`InsightFace` 提供了多种检测器，如 `RetinaFace` 和 `SCRFD`。

-   **RetinaFace**: 一个强大的多任务检测器，能同时输出人脸边界框、5个关键点以及3D姿态。
-   **SCRFD**: 一种高效且高精度的检测器，在各种硬件平台上都有出色的表现。

通过 `app.get(image)` 方法，我们可以轻松地从图像中获取人脸信息。

```python
import cv2
import numpy as np

# 读取图片
img = cv2.imread("./data/test/t1.jpg")

# 获取人脸信息
faces = app.get(img)

# 遍历检测到的所有人脸并绘制
for face in faces:
    # 绘制边界框
    bbox = face.bbox.astype(np.int32)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # 绘制106个关键点
    landmarks = face.landmark_2d_106.astype(np.int32)
    for landmark in landmarks:
        cv2.circle(img, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)

# 显示或保存结果
cv2.imwrite("test_out/t1_output.jpg", img)
```

`app.get()` 返回一个 `Face` 对象列表，每个 `Face` 对象包含了丰富的人脸属性，例如：

*   `bbox`: 人脸边界框 `[x1, y1, x2, y2]`
*   `kps`: 5个关键点坐标
*   `landmark_2d_106`: 106个面部关键点坐标
*   `embedding`: 512维的人脸特征向量（用于人脸识别）
*   `gender`: 性别 (0: 女性, 1: 男性)
*   `age`: 预估年龄

**效果示例:**

 原始图片及检测效果 

| ![1](https://github.com/user-attachments/assets/ab042f7c-6255-42a9-bab0-8dfd36feaa6a)
| ![1](https://github.com/user-attachments/assets/5209b7ec-4c19-43fe-b8b5-145bd7671ada)
 
|<img width="1067" height="924" alt="7" src="https://github.com/user-attachments/assets/1d7471de-13fa-47ba-acd8-017ac1b3bcf4" />
| <img width="1067" height="924" alt="7" src="https://github.com/user-attachments/assets/954cb82f-64bb-4837-bd5d-713665eb71d4" />
|

## 2. 人脸识别

人脸识别的核心是比较两个人脸特征向量的相似度。`InsightFace` 提供了 `ArcFace`, `SubCenter-ArcFace` 等先进的识别模型，可以将人脸图片转换成一个 512 维的特征向量 (`embedding`)。

#### 基本流程

1.  **建立人脸数据库**: 将已知身份的人脸图片转换成特征向量，并与身份（如姓名）关联起来，存储备用。
2.  **提取待识别人脸的特征**: 获取摄像头或图片中新人脸的特征向量。
3.  **计算相似度**: 将新的人脸特征向量与数据库中所有特征向量进行比较。
4.  **判定身份**: 如果相似度得分高于预设阈值，则认为匹配成功。

#### 2.1 建立人脸数据库

我们可以创建一个文件夹（例如 `data/faces`），每张图片以人名命名（如 `成毅.png`, `虞书欣.jpg`）。然后编写一个函数来加载这些图片并提取特征。

```python
import os
import numpy as np
from sklearn import preprocessing

# 加载人脸数据库，返回一个包含姓名和对应特征向量的字典
def load_face_embeddings(face_folder):
    face_dict = {}
    for file in os.listdir(face_folder):
        name = os.path.splitext(file)[0]  # 使用文件名作为人名
        img_path = os.path.join(face_folder, file)
        img = cv2.imread(img_path)
        
        # 获取人脸信息，这里假设每张注册照只有一张脸
        face = app.get(img)[0] 
        
        # 提取特征向量并进行归一化
        embedding = np.array(face.embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)
        face_dict[name] = embedding
    return face_dict

# 加载数据库
face_db = load_face_embeddings('./data/faces')
print(f"人脸数据库加载完毕，共 {len(face_db)} 人。")
```

#### 2.2 特征向量比对

比较两个特征向量通常使用欧氏距离或余弦相似度。`InsightFace` 官方推荐使用欧氏距离的平方和。距离越小，代表两个人脸越相似。

```python
# 计算两个特征向量的距离
def feature_compare(feature1, feature2):
    diff = np.subtract(feature1, feature2)
    dist = np.sum(np.square(diff), 1)
    return dist
```

#### 2.3 实时人脸识别（摄像头）

结合以上步骤，我们可以实现一个通过摄像头进行实时人脸识别的程序。

```python
# 封装识别逻辑
def recognition(faces, face_db, threshold=1.3):
    results = []
    for face in faces:
        # 提取当前帧人脸的特征向量并归一化
        embedding = np.array(face.embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)
        
        best_name = 'Unknown'
        best_score = float('inf')
        
        # 与数据库中的每一个人脸进行比对
        for name, db_embedding in face_db.items():
            score = feature_compare(embedding, db_embedding)
            if score < best_score:
                best_score = score
                best_name = name
        
        # 如果最小距离小于阈值，则认为是同一个人
        if best_score < threshold:
            results.append({"name": best_name, "score": best_score, "bbox": face.bbox})
        else:
            results.append({"name": "Unknown", "score": best_score, "bbox": face.bbox})
    return results

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 检测人脸
    faces = app.get(frame)
    
    # 进行识别
    rec_results = recognition(faces, face_db)
    
    # 在画面上绘制结果
    for res in rec_results:
        bbox = res['bbox'].astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # 显示姓名和相似度得分
        text = f"{res['name']} ({res['score'][0]:.2f})"
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
#### 2.4成毅测试结果
![3](https://github.com/user-attachments/assets/fbc0b675-690d-43bf-a49e-ccf9735cd78d)
<img width="1078" height="721" alt="6" src="https://github.com/user-attachments/assets/9e9440e5-3237-4f25-81ee-7825db3d9d3f" />
可以看出 模型可以准确识别出成毅并在正确的脸上进行标注
## 3. 一些测试结果
![11_out](https://github.com/user-attachments/assets/dd677a8b-9cf0-407a-87fc-654c7ba5227c)
![12_out](https://github.com/user-attachments/assets/9040bb6a-fc28-4988-9f0c-f9f01dd12128)
![15_out](https://github.com/user-attachments/assets/f723bdd2-c1af-4e3d-af7b-1aaa7ff5d31e)
![9_out](https://github.com/user-attachments/assets/acd7439b-2fb2-4ae0-8310-02a61bbbc94e)
![3_out](https://github.com/user-attachments/assets/11dace88-21d2-45b9-a139-e98f0f651bd0)
![4_out](https://github.com/user-attachments/assets/db7869b6-1619-4d47-a151-931d3ba1ea58)
![7_out](https://github.com/user-attachments/assets/b2f4177f-c27e-44a6-a209-4eb9f1b0f0d2)


