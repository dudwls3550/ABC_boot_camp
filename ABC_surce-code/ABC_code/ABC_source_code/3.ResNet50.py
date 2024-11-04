from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ResNet50 모델을 불러옵니다
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 레이어 동결
for layer in base_model.layers:
    layer.trainable = False

# 새로운 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 데이터셋 로드 및 증강
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'augmented_faces',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'augmented_faces',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 모델 학습
model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)

# 모델 저장
model.save('vggface_finetuned_model.h5')
