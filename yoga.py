import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load the model and labels
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Header
st.header("Emotion-Based Yoga Recommender")

# Session state to manage flow
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load saved emotion if available
try:
    emotion = np.load("emotion.npy")[0]
except FileNotFoundError:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Class for processing emotion
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Flip frame for a mirror effect
        frm = cv2.flip(frm, 1)

        # Process the frame with Mediapipe
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            # Predict emotion
            pred = label[np.argmax(model.predict(lst))]
            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            # Save emotion
            np.save("emotion.npy", np.array([pred]))

        # Draw landmarks
        drawing.draw_landmarks(
            frm,
            res.face_landmarks,
            holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
            connection_drawing_spec=drawing.DrawingSpec(thickness=1),
        )
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# Input fields
lang = st.text_input("Preferred Language")

# Dropdown menus for category and type of yoga
category = st.selectbox(
    "Select Age Category",
    ["Children (Below 12)", "Teenagers", "Adults", "Seniors"]
)
yoga_type = st.selectbox(
    "Select Type of Yoga",
    ["Beginner Yoga", "Intermediate Yoga", "Advanced Yoga", "Relaxation Yoga", "Therapeutic Yoga"]
)

# Webcam streamer
if lang and yoga_type and st.session_state["run"] != "false":
    webrtc_streamer(
        key="key",
        desired_playing_state=True,
        video_processor_factory=EmotionProcessor,
    )

# Recommend button
btn = st.button("Recommend me")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first.")
        st.session_state["run"] = "true"
    else:
        # Add additional search terms based on dropdowns
        search_query = f"{emotion} {yoga_type} for {category} in {lang} "
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"