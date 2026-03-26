"""
Test your own trained model in real time
Remember/keep it in mind Run this AFTER train_model.py has finished
"""

import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
import os


# Paths
MODEL_PATH   = r"F:\python\sign_language_detection\hand_landmarker.task"
MY_MODEL     = r"F:\python\sign_language_detection\my_model.h5"
LABELS_FILE  = r"F:\python\sign_language_detection\my_labels.txt"

# Load your trained model and labels
model  = load_model(MY_MODEL, compile=False)
labels = [line.strip() for line in open(LABELS_FILE).readlines()]
print("Labels:", labels)

# Load hand detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector     = vision.HandLandmarker.create_from_options(options)

cap     = cv2.VideoCapture(0)
offset  = 20
imgSize = 300

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

while True:
    success, img = cap.read()
    if not success:
        break

    imgH, imgW  = img.shape[:2]
    img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result      = detector.detect(mp_image)

    imgWhite         = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    prediction_text  = ""

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        pts = [(int(lm.x * imgW), int(lm.y * imgH)) for lm in lms]

        # Draw skeleton
        for a, b in CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)
        for pt in pts:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)

        # Bounding box
        xs, ys = zip(*pts)
        x1 = max(0, min(xs) - offset)
        y1 = max(0, min(ys) - offset)
        x2 = min(imgW, max(xs) + offset)
        y2 = min(imgH, max(ys) + offset)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size != 0:
            h, w = imgCrop.shape[:2]
            if h / w > 1:
                k    = imgSize / h
                wCal = min(math.ceil(k * w), imgSize)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k    = imgSize / w
                hCal = min(math.ceil(k * h), imgSize)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Predict
            img_input   = cv2.resize(imgWhite, (224, 224)).astype("float32") / 255.0
            img_input   = np.expand_dims(img_input, axis=0)
            predictions = model.predict(img_input, verbose=0)
            idx         = np.argmax(predictions)
            confidence  = predictions[0][idx] * 100
            prediction_text = f"{labels[idx]}  ({confidence:.1f}%)"

            cv2.putText(img, prediction_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Combined window
    imgWhiteResized = cv2.resize(imgWhite, (imgSize, imgH))
    combined = np.hstack([img, imgWhiteResized])
    cv2.putText(combined, prediction_text if prediction_text else "Show your hand...",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(combined, "Q = Quit", (10, imgH - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Sign Language Detection - My Model", combined)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
