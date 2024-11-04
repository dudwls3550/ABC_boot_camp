import cv2
import numpy as np
import face_recognition
import csv
import datetime
import pickle

# 얼굴 인코딩 파일 로드
def load_encodings(encodings_file):
    with open(encodings_file, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

# 인코딩 파일 경로
encodings_file = 'encodings_final.pkl'

# 얼굴 데이터베이스 로드 및 인코딩
known_face_encodings, known_face_names = load_encodings(encodings_file)

# CSV 파일 설정
csv_file = ('renewal_recognition_log.csv')

# CSV 파일에 헤더 추가 (파일이 존재하지 않을 때만 추가)
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Entry Time", "Exit Time"])

# 사람들의 현재 상태를 추적하기 위한 딕셔너리
person_times = {}
active_persons = set()

def log_entry_exit(name, entry_time, exit_time):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, entry_time, exit_time])

# 웹캠을 통해 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_frame_persons = set()
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        # face_distance 계산
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < 0.35:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
        else:
            name = "Unknown"
        current_frame_persons.add(name)

        if name != "Unknown" and name not in active_persons:
            entry_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            person_times[name] = entry_time
            active_persons.add(name)

    for name in active_persons.copy():
        if name not in current_frame_persons:
            exit_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            log_entry_exit(name, person_times[name], exit_time)
            active_persons.remove(name)
            del person_times[name]

    for (top, right, bottom, left), name in zip(face_locations, current_frame_persons):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name , (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
