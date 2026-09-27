import os
import cv2
import pickle
from cvzone.HandTrackingModule import HandDetector

DATA_DIR = './data'

data = []
labels = []

# 1. Use staticMode=True for independent dataset images
# Set maxHands=1 to maintain a consistent 84-feature length per sample
detector = HandDetector(staticMode=True, maxHands=1, detectionCon=0.5)

# Loop through folder names '0' to '35'
for i in range(36):
    dir_name = str(i)
    dir_path = os.path.join(DATA_DIR, dir_name)

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        continue

    print(f"Processing class folder: {dir_name}")

    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)
        if img is None:
            continue

        # 2. Disable drawing (draw=False) to maximize processing speed
        hands, _ = detector.findHands(img, draw=False)

        if hands:
            for hand in hands:
                lmList = hand['lmList']
                data_aux = []

                x_vals = [lm[0] for lm in lmList]
                y_vals = [lm[1] for lm in lmList]

                # Only append normalized/relative coordinates (42 features total)
                for lm in lmList:
                    data_aux.extend([lm[0] - min(x_vals), lm[1] - min(y_vals)])

                data.append(data_aux)
                labels.append(dir_name)

# Save dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Done! Successfully processed {len(data)} hand samples.")