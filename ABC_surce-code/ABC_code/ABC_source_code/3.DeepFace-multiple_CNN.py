from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def check_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            try:
                img = Image.open(os.path.join(directory, filename))
                img.verify()  # 이미지 파일을 검증
            except (IOError, SyntaxError) as e:
                print(f"Bad file: {filename}")
                os.remove(os.path.join(directory, filename))  # 문제가 있는 파일 제거

# 특정 인물 및 unknown 데이터 경로 확인
for person in ["wootae", "yeongjin","hayeon","jieun","minju","rulwon"]:
    person_dir = os.path.join('augmented_faces', person)
    check_images(person_dir)



# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # 20% validation data

train_generator = train_datagen.flow_from_directory(
    'augmented_faces',  # 데이터셋 폴더 경로
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training')  # training data

validation_generator = train_datagen.flow_from_directory(
    'augmented_faces',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation')  # validation data

# tf.data API를 사용하여 데이터셋 생성
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 224, 224, 3], [None, 6])
)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 224, 224, 3], [None, 6])
)

# Prefetch, shuffle, and batch operations
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)




# 다중 CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 모델 학습
history = model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_dataset,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# 모델 저장
model.save('multiple_CNN_model.h5')
