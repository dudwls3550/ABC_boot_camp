import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sort import Sort
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine

# 모델 로드
model = load_model('multiple_CNN_model.h5')

# 얼굴 임베딩 데이터베이스 로드
database = np.load('multiple_CNN_embeddings.npy', allow_pickle=True).item()


# 얼굴 임베딩을 얻는 함수
def get_embedding(model, face):
    face = cv2.resize(face, (224, 224))
    face = np.expand_dims(face, axis=0)
    face = face / 255.0
    embedding = model.predict(face)
    return embedding.flatten()  # 1차원으로 변환


# 가장 유사한 얼굴을 찾는 함수
def find_best_match(embedding, database, threshold=0.5):
    min_dist = float('inf')
    best_match = "unknown"
    for name, db_embedding in database.items():
        dist = cosine(embedding, db_embedding.flatten())  # db_embedding도 1차원으로 변환
        if dist < min_dist:
            min_dist = dist
            best_match = name

    # 정확도를 계산 (1 - cosine distance)
    accuracy = max(0, 1 - min_dist)

    # 만약 거리가 threshold보다 크면 "unknown"으로 처리
    if min_dist >= threshold:
        best_match = "unknown"
        accuracy = 0.0

    return best_match, accuracy


# 웹캠을 통해 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

# 시간 기록을 위한 데이터 초기화
time_data = []

tracker = Sort()


# 시간 기록 데이터를 DataFrame으로 변환 및 저장하는 함수
def save_time_data(time_data):
    time_df = pd.DataFrame(time_data, columns=['Name', 'Enter_Time', 'Exit_Time'])
    time_df.to_csv("time_records.csv", index=False)


# 얼굴 인식 및 시간 기록
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 탐지 (여기서는 Haar Cascade를 사용합니다. 다른 방법을 사용할 수 있습니다.)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_faces = []

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        embedding = get_embedding(model, face)

        # 얼굴 인식
        name, accuracy = find_best_match(embedding, database)

        # SORT로 트래킹 박스 업데이트
        detections = [[x, y, x + w, y + h, 1.0]]
        tracked_objects = tracker.update(np.array(detections))

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track.astype(int)
            current_faces.append(name)

            # 출입 시간 기록
            if name != "unknown":
                # 사람이 처음 들어왔을 때 기록
                if name not in [entry[0] for entry in time_data]:
                    enter_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    time_data.append([name, enter_time, None])
                    print(f"{name} entered at {enter_time}")
                    save_time_data(time_data)  # 실시간으로 저장
                # 사람이 나갈 때 기록
                elif name in [entry[0] for entry in time_data if entry[2] is None]:
                    for entry in time_data:
                        if entry[0] == name and entry[2] is None:
                            exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            entry[2] = exit_time
                            print(f"{name} exited at {exit_time}")
                            save_time_data(time_data)  # 실시간으로 저장

            # 시각화
            label = f"{name} ({accuracy * 100:.2f}%)"  # 이름과 정확도를 함께 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 프로그램 종료 시 마지막으로 시간 기록 데이터를 저장
save_time_data(time_data)
