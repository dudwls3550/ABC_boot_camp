import os
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# 모델 로드
model = load_model('multiple_CNN_model.h5')

# 얼굴 임베딩을 얻는 함수
def get_embedding(model, face):
    face = cv2.resize(face, (224, 224))
    face = np.expand_dims(face, axis=0)
    face = face / 255.0
    embedding = model.predict(face)
    return embedding

# 각 클래스에 대한 임베딩 저장
database = {}
classes = ["hayeon", "jieun","minju","rulwon","wootae","yeongjin"]
for cls in classes:
    class_dir = os.path.join('augmented_faces', cls)
    for filename in os.listdir(class_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            embedding = get_embedding(model, img)
            database[cls] = embedding
            break  # 각 클래스당 하나의 대표 이미지 사용

np.save('multiple_CNN_embeddings.npy', database)
print("Face embeddings have been saved.")
