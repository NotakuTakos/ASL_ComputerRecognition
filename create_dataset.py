import os
import cv2
import pickle
from cvzone.HandTrackingModule import HandDetector

# Hand skeleton connections (21 hand landmarks: 0 through 20)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]

detector = HandDetector(staticMode=True, maxHands=2, detectionCon=0.3)
DATA_DIR = './data'

data = []
labels = []

# Loop through folder names '0' to '35'
for i in range(36):
    dir_name = str(i)
    dir_path = os.path.join(DATA_DIR, dir_name)

    # Skip if the target class directory does not exist
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        data_aux = []

        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)
        if img is None:
            continue

        hands, img_drawn = detector.findHands(img, draw=True)

        if hands:
            for hand in hands:
                lmList = hand['lmList']  # List of 21 landmark points

                for i, lm in enumerate(lmList):
                    x, y = lm[0], lm[1]

                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_name)



                # 1. Draw connecting lines ("bones")
                #for p1, p2 in HAND_CONNECTIONS:
                 #   x1, y1 = lmList[p1][0], lmList[p1][1]
                  #  x2, y2 = lmList[p2][0], lmList[p2][1]
                   # cv2.line(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 2. Draw joint dots
                #for lm in lmList:
                 #   cx, cy = lm[0], lm[1]
                  #  cv2.circle(img_drawn, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

        img_rgb = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()