from PIL import Image, ImageEnhance, ImageOps
import random
import os


def augment_image(image_path, output_dir, num_augmentations=5):
    image = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(num_augmentations):
        # 회전
        angle = random.uniform(-30, 30)
        rotated_image = image.rotate(angle)

        # 밝기 조절
        enhancer = ImageEnhance.Brightness(rotated_image)
        factor = random.uniform(0.01, 1)
        bright_image = enhancer.enhance(factor)

        # 대칭
        if random.random() > 0.5:
            bright_image = ImageOps.mirror(bright_image)

        # 저장
        output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.png")
        bright_image.save(output_path)
        print(f"Saved augmented image: {output_path}")


# 예시 사용
input_image_dir = 'augmented_faces/hayeon'
output_dir = 'augmented_faces/hayeon'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        augment_image(os.path.join(input_image_dir, filename), output_dir, num_augmentations=150)


input_image_dir = 'augmented_faces/jieun'
output_dir = 'augmented_faces/jieun'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        augment_image(os.path.join(input_image_dir, filename), output_dir, num_augmentations=150)



input_image_dir = 'augmented_faces/minju'
output_dir = 'augmented_faces/minju'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        augment_image(os.path.join(input_image_dir, filename), output_dir, num_augmentations=150)




input_image_dir = 'augmented_faces/rulwon'
output_dir = 'augmented_faces/rulwon'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        augment_image(os.path.join(input_image_dir, filename), output_dir, num_augmentations=150)


input_image_dir = 'augmented_faces/wootae'
output_dir = 'augmented_faces/wootae'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        augment_image(os.path.join(input_image_dir, filename), output_dir, num_augmentations=150)


input_image_dir = 'augmented_faces/yeongjin'
output_dir = 'augmented_faces/yeongjin'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        augment_image(os.path.join(input_image_dir, filename), output_dir, num_augmentations=150)

