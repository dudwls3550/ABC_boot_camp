from sklearn.model_selection import train_test_split
import pandas as pd
import os

# 특정 인물 및 "unknown" 데이터를 준비합니다.
data = []
labels = []

# 특정 인물 데이터 로드
for person in ["wootae", "yeongjin","hayeon","jieun","minju","rulwon"]:
    person_dir = os.path.join('augmented_faces', person)
    for filename in os.listdir(person_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            data.append(os.path.join(person_dir, filename))
            labels.append(person)



# 데이터를 DataFrame으로 변환
df = pd.DataFrame({"image": data, "label": labels})

# 학습 데이터와 테스트 데이터로 분할
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)

print("Data has been split and saved to 'train_data.csv' and 'test_data.csv'")
print(f"Training data size: {len(train_df)}")
print(f"Testing data size: {len(test_df)}")
