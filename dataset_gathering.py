import cv2
import numpy as np
import math
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = r"F:\python\sign_language_detection\hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
counter = 0
folder = r"F:\python\sign_language_detection\data\Thank you"
os.makedirs(folder, exist_ok=True)

# Hand connections for drawing skeleton
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20), # Pinky
    (5,9),(9,13),(13,17)            # Palm
]

while True:
    success, img = cap.read()
    if not success:
        break

    imgH, imgW = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        pts = [(int(lm.x * imgW), int(lm.y * imgH)) for lm in lms]

        # Draw skeleton lines
        for a, b in CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)

        # Draw landmark dots
        for pt in pts:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)

        # Bounding box
        xs, ys = zip(*pts)
        x1 = max(0, min(xs) - offset)
        y1 = max(0, min(ys) - offset)
        x2 = min(imgW, max(xs) + offset)
        y2 = min(imgH, max(ys) + offset)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Hand label
        if result.handedness:
            label = result.handedness[0][0].display_name
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        # Crop and fit into white square
        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size != 0:
            h, w = imgCrop.shape[:2]
            if h / w > 1:
                k = imgSize / h
                wCal = min(math.ceil(k * w), imgSize)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = min(math.ceil(k * h), imgSize)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

    # Single combined window
    imgWhiteResized = cv2.resize(imgWhite, (imgSize, imgH))
    combined = np.hstack([img, imgWhiteResized])
    cv2.putText(combined, f"Saved: {counter}  |  S=Save  Q=Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Sign Language Collection", combined)

    key = cv2.waitKey(1)
    if key == ord("s") and result.hand_landmarks:
        counter += 1
        cv2.imwrite(os.path.join(folder, f"Image_{time.time()}.jpg"), imgWhite)
        print(f"Saved: {counter}")
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()