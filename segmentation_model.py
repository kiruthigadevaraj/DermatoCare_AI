import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Image size
IMG_SIZE = 128

# Folder paths
image_dir = "Images"
mask_dir = "Masks"

images = []
masks = []

print("Loading data...")

# Get all image files
image_files = os.listdir(image_dir)

for file in image_files:

    img_path = os.path.join(image_dir, file)

    # Remove extension properly
    filename_without_ext = os.path.splitext(file)[0]
    mask_name = filename_without_ext + "_segmentation.png"
    mask_path = os.path.join(mask_dir, mask_name)

    if os.path.exists(mask_path):

        # Load image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        # Load mask (grayscale)
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    else:
        print("Mask not found for:", file)

images = np.array(images)
masks = np.array(masks)

print("Data loaded successfully!")
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)

# ======================
# SIMPLE U-NET MODEL
# ======================

def build_unet():

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b1 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(b1)
    c3 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D((2, 2))(c3)
    c4 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    model = models.Model(inputs, outputs)
    return model


model = build_unet()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Training started...")

model.fit(images, masks, epochs=5, batch_size=8)

model.save("segmentation_model.h5")

print("Model saved successfully!")