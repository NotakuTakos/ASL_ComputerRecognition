import os
import cv2
import mediapipe as mp

# 1. Initialize MediaPipe solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    print(f"Error: Directory '{DATA_DIR}' not found.")
    exit()

# 2. Iterate through class folders
for dir_name in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_name)
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path)[:1]:
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue

        # Convert BGR to RGB for MediaPipe inference
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Draw landmarks onto original BGR image for OpenCV display
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Reuses the SAME window titled "ASL Dataset Inspection"
        cv2.imshow("ASL Dataset Inspection", img)
        cv2.waitKey(0)  # Press ANY key to view the next image

cv2.destroyAllWindows()
hands.close()