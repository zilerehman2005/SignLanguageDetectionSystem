"""
Train your own model using TensorFlow/Keras
Run this ONCE to train. It will save my_model.h5 and labels.txt
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Config
DATA_DIR   = r"F:\python\sign_language_detection\data"
SAVE_MODEL = r"F:\python\sign_language_detection\my_model.h5"
SAVE_LABELS= r"F:\python\sign_language_detection\my_labels.txt"
IMG_SIZE   = 224

# Step 1 - Load all images from each class folder
images = []
labels_list = []
class_names = sorted(os.listdir(DATA_DIR))  # ['Hello', 'I Love You', 'Thank You', 'Yes']

print("Classes found:", class_names)

for idx, class_name in enumerate(class_names):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    files = os.listdir(class_path)
    print(f"  Loading {class_name}: {len(files)} images")
    for file in files:
        img_path = os.path.join(class_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        images.append(img)
        labels_list.append(idx)

X = np.array(images)
y = to_categorical(np.array(labels_list), num_classes=len(class_names))

print(f"\nTotal images: {len(X)}, Classes: {len(class_names)}")

# Step 2 - Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Step 3 - Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Step 4 - Train
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Step 5 - Evaluate and save
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

model.save(SAVE_MODEL)
print(f"Model saved to: {SAVE_MODEL}")

with open(SAVE_LABELS, "w") as f:
    for name in class_names:
        f.write(name + "\n")
print(f"Labels saved to: {SAVE_LABELS}")

# Step 6 - Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(r"F:\python\sign_language_detection\training_plot.png")
plt.show()
print("Training complete!")
