import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return [input_tensor * self.scale for input_tensor in inputs]
        else:
            return inputs * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({"scale": self.scale})
        return config

# CSV 파일 로드
test_df = pd.read_csv('test_df.csv')

# 이미지 로드 및 전처리 함수 정의
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = img / 255.0
    return img

# 테스트 데이터 준비
X_test = np.array([load_and_preprocess_image(img_path) for img_path in test_df['image'].values])

# 클래스 이름과 인덱스 매핑
class_indices = {class_name: index for index, class_name in enumerate(sorted(test_df['label'].unique()))}

# y_true를 숫자 인덱스로 변환
y_true = np.array([class_indices[label] for label in test_df['label'].values])

# 모델 파일 경로
model_paths = [
    'Inception-ResNet_model.h5',
    'ResNet_model.h5',
    'vgg16_model.h5',
    'simple_CNN_model.h5',
    'multiple_CNN_model.h5'
]

# 성능 지표 저장
model_names = ['Inception-ResNet','ResNet','VGG16', 'Simple CNN', 'Multiple CNN']
accuracies = []
precisions = []
recalls = []
f1_scores = []

# 각 모델에 대해 예측 및 성능 평가 수행
for model_path in model_paths:
    with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
        model = tf.keras.models.load_model(model_path)
    predictions = model.predict(X_test)
    y_pred = predictions.argmax(axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# 성능 지표 출력
for name, acc, prec, rec, f1 in zip(model_names, accuracies, precisions, recalls, f1_scores):
    print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1 Score={f1:.4f}")

# 시각화 (수평 바 차트)
x = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

# 각 성능 지표에 대한 막대 색상 설정
colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99']

# 수평 막대 그래프 생성
rects1 = ax.barh(x - 1.5*width, accuracies, width, label='Accuracy', color=colors[0])
rects2 = ax.barh(x - 0.5*width, precisions, width, label='Precision', color=colors[1])
rects3 = ax.barh(x + 0.5*width, recalls, width, label='Recall', color=colors[2])
rects4 = ax.barh(x + 1.5*width, f1_scores, width, label='F1 Score', color=colors[3])

# 그래프에 추가 정보
ax.set_ylabel('Models')
ax.set_xlabel('Scores')
ax.set_title('Comparison of Model Performance')
ax.set_yticks(x)
ax.set_yticklabels(model_names)
ax.legend()

# 레이블 추가
def add_labels(rects):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.4f}',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
add_labels(rects4)

plt.show()
