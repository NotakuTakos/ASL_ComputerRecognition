import streamlit as st
import cv2
import pickle
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# 1. Setup Streamlit UI
st.set_page_config(page_title="Sign Language Detector", layout="wide")
st.title("Sign Language Recognition")
st.write("Turn on the webcam to start detecting sign language.")

# 2. Cache resources to prevent duplicate C++ thread creation
@st.cache_resource
def load_model():
    model_dict = pickle.load(open('./model.p', 'rb'))
    return model_dict['model']

@st.cache_resource
def load_detector():
    return HandDetector(staticMode=False, maxHands=1, detectionCon=0.3)

model = load_model()
detector = load_detector()

# Constants
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
               8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
               16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
               25: 'Z', 26: '1', 27: '2', 28: '3', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8',
               34: '9', 35: '0'}

# 3. Create controls
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

# 4. Open and manage camera with guaranteed cleanup
if run:
    cap = cv2.VideoCapture(0)

    try:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video from webcam.")
                break

            # Process the frame
            hands, img_drawn = detector.findHands(frame, draw=False)

            if hands:
                for hand in hands:
                    lmList = hand['lmList']
                    data_aux = []
                    x_vals = [lm[0] for lm in lmList]
                    y_vals = [lm[1] for lm in lmList]

                    for lm in lmList:
                        data_aux.extend([lm[0], lm[1]])

                    for lm in lmList:
                        data_aux.extend([lm[0] - min(x_vals), lm[1] - min(y_vals)])

                    for p1, p2 in HAND_CONNECTIONS:
                        x1, y1 = lmList[p1][0], lmList[p1][1]
                        x2, y2 = lmList[p2][0], lmList[p2][1]
                        cv2.line(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    for lm in lmList:
                        cx, cy = lm[0], lm[1]
                        cv2.circle(img_drawn, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    x_min, y_min = min(x_vals), min(y_vals)
                    cv2.putText(img_drawn, predicted_character, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            img_rgb = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(img_rgb)

    finally:
        # GUARANTEED CLEANUP: Executes whether the user unchecks the box,
        # an exception occurs, or Ctrl+C is pressed.
        cap.release()
        cv2.destroyAllWindows()
        FRAME_WINDOW.empty()

else:
    st.info("Click the checkbox to start the webcam feed.")