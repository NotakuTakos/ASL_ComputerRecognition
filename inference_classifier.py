import streamlit as st
import cv2
import pickle
import numpy as np
import av
from cvzone.HandTrackingModule import HandDetector
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# 1. Setup Streamlit UI
st.set_page_config(page_title="Sign Language Detector", layout="wide")
st.title("Sign Language Recognition")
st.write("Grant camera permissions to start detecting sign language.")

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

# 2. Define the WebRTC Video Processor
# In Streamlit Cloud, video processing happens in a separate background thread.
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        # Load models inside the init function so they exist in the correct thread
        model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = model_dict['model']
        self.detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.3)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the browser's video frame to an OpenCV readable array
        img = frame.to_ndarray(format="bgr24")

        # Process the frame
        hands, img_drawn = self.detector.findHands(img, draw=False)

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

                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                x_min, y_min = min(x_vals), min(y_vals)
                cv2.putText(img_drawn, predicted_character, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Return the annotated frame back to the browser
        return av.VideoFrame.from_ndarray(img_drawn, format="bgr24")


# 3. WebRTC Configuration
# STUN servers help the cloud app punch through firewalls to reach the user's webcam
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# 4. Start the Stream
webrtc_streamer(
    key="sign_language",
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)