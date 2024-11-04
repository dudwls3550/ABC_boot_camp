import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# CSV 파일 로드
test_df = pd.read_csv('test_df.csv')

# 이미지 로드 및 전처리
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # 이미지 로드 및 크기 조정
    img = img_to_array(img)  # 이미지를 numpy array로 변환
    img = img / 255.0  # 스케일링
    return img

# 테스트 데이터 준비
X_test = np.array([load_and_preprocess_image(img_path) for img_path in test_df['image'].values])

# 클래스 이름과 인덱스 매핑
class_indices = {class_name: index for index, class_name in enumerate(sorted(test_df['label'].unique()))}

# y_true를 숫자 인덱스로 변환
y_true = np.array([class_indices[label] for label in test_df['label'].values])

# 모델 로드
model = tf.keras.models.load_model('Inception-ResNet_model.h5')

# 예측 수행
predictions = model.predict(X_test)
y_pred = predictions.argmax(axis=1)

# 성능 평가
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n {conf_matrix}")
