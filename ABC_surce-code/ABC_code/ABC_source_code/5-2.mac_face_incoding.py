import face_recognition
import os
import pickle

# 얼굴 이미지가 저장된 디렉토리 경로
KNOWN_FACES_DIR = 'final_face'

# 얼굴 인코딩 생성 및 저장
def create_encodings_and_save():
    known_faces = []
    known_names = []
    encoding_count = {}  # 성공적으로 인코딩된 이미지 수
    total_count = {}     # 처리된 총 이미지 수

    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_dir):
            print(f"Processing {name}...")
            encoding_count[name] = 0
            total_count[name] = 0  # 초기화

            for filename in os.listdir(person_dir):
                if filename == '.DS_Store':
                    continue

                image_path = os.path.join(person_dir, filename)
                print(f"  Encoding {filename}...")

                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                total_count[name] += 1  # 이미지 처리 카운트 증가

                if encodings:
                    known_faces.append(encodings[0])
                    known_names.append(name)
                    encoding_count[name] += 1  # 인코딩 성공 카운트 증가
                    print(f"    Successfully encoded {filename}.")
                else:
                    print(f"    Failed to encode {filename} (no faces found).")

    # 인코딩과 이름을 저장
    with open('encodings_final.pkl', 'wb') as f:
        pickle.dump((known_faces, known_names), f)
    print("Encodings saved successfully!")

    # 각 이름에 대해 인코딩된 이미지 수와 인코딩 성공률 출력
    for name in encoding_count:
        success_rate = encoding_count[name] / total_count[name] if total_count[name] > 0 else 0
        print(f"{name}: {encoding_count[name]} images encoded out of {total_count[name]} processed. Success rate: {success_rate:.2f}")

create_encodings_and_save()
