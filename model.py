import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 128

# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset():
    print("Reading CSV file...")
    data = pd.read_csv("dataset/labels.csv")

    images = []
    labels = []

    print("Loading images...")
    for index, row in data.iterrows():
        image_id = row['image_id']
        label = row['dx']

        image_path = os.path.join("dataset/images", image_id + ".jpg")

        if os.path.exists(image_path):
            img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)

    images = np.array(images)

    print("Encoding labels...")
    unique_labels = list(set(labels))
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_dict[label] for label in labels]
    numeric_labels = to_categorical(numeric_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        images, numeric_labels, test_size=0.2, random_state=42
    )

    print("Dataset ready!")
    return X_train, X_test, y_train, y_test, len(unique_labels)

# -----------------------------
# BUILD MODEL
# -----------------------------
def build_model(num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model():
    print("Loading dataset...")
    X_train, X_test, y_train, y_test, num_classes = load_dataset()

    print("Building model...")
    model = build_model(num_classes)

    print("Training started...")
    model.fit(X_train, y_train, epochs=5, batch_size=32,
              validation_data=(X_test, y_test))

    print("Saving model...")
    model.save("skin_cancer_model.h5")

    print("Training completed and model saved successfully!")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    train_model()