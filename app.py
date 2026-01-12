import streamlit as st
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer
import av

# ============================
# Database imports
# ============================
from database import (
    initialize_database,
    save_detection,
    fetch_detections,
    clear_detections,
    add_user,
    verify_user
)

# ============================
# Streamlit Config
# ============================
st.set_page_config(page_title="Anti-Poaching System", layout="wide")

# ============================
# Twilio (Environment Variables)
# ============================
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")

client = Client(TWILIO_SID, TWILIO_AUTH) if TWILIO_SID else None

def send_sms_alert(message):
    if client is None:
        return None
    try:
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        return msg.sid
    except Exception:
        return None

# ============================
# Load YOLO Model (Cached)
# ============================
MODEL_PATH = "runs/detect/train11/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ============================
# Init Database
# ============================
initialize_database()

# ============================
# Session State
# ============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = None

# ============================
# Sidebar Menu
# ============================
st.sidebar.title("Navigation")

menu = st.sidebar.selectbox(
    "Menu",
    ["Login", "Sign Up"] if not st.session_state.authenticated
    else ["Home", "Dashboard", "Database", "Heat Map", "Logout"]
)

# ============================
# LOGIN
# ============================
if menu == "Login" and not st.session_state.authenticated:
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(u, p):
            st.session_state.authenticated = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

# ============================
# SIGN UP
# ============================
elif menu == "Sign Up" and not st.session_state.authenticated:
    st.subheader("📝 Create Account")
    u = st.text_input("New Username")
    p1 = st.text_input("Password", type="password")
    p2 = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if p1 == p2:
            add_user(u, p1)
            st.success("Account created")
        else:
            st.error("Passwords do not match")

# ============================
# LOGOUT
# ============================
elif menu == "Logout":
    st.session_state.authenticated = False
    st.stop()

# ============================
# HOME
# ============================
elif menu == "Home" and st.session_state.authenticated:
    st.title("🦌 Anti-Poaching Detection System")
    st.markdown("""
    **Features**
    - Live Camera Detection (Browser Webcam)
    - Video Upload Detection
    - YOLO-based Poacher Detection
    - SMS Alerts
    - Database Storage
    - Weekly Heatmap Analysis
    """)

# ============================
# DASHBOARD
# ============================
elif menu == "Dashboard" and st.session_state.authenticated:
    st.title("🎥 Detection Dashboard")

    POACHER_CLASS_ID = 0
    ALERT_COOLDOWN = 60  # seconds

    def handle_detection(annotated, count):
        if count <= 0:
            return

        now = datetime.now()

        if (
            st.session_state.last_alert_time is None or
            (now - st.session_state.last_alert_time).seconds > ALERT_COOLDOWN
        ):
            send_sms_alert(
                f"🚨 Poacher detected at {now.strftime('%Y-%m-%d %H:%M:%S')} | Count: {count}"
            )

            _, img_enc = cv2.imencode(".jpg", annotated)
            save_detection(
                now.strftime('%Y-%m-%d %H:%M:%S'),
                count,
                img_enc.tobytes()
            )

            st.session_state.last_alert_time = now

    mode = st.radio("Select Mode", ["Live Camera", "Upload Video"])

    # ---------- LIVE CAMERA (WebRTC) ----------
    if mode == "Live Camera":

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")

            results = model.predict(img, conf=0.5, device="cpu", verbose=False)
            boxes = results[0].boxes
            detected = boxes.cls.cpu().numpy() if boxes else []
            count = int((detected == POACHER_CLASS_ID).sum())

            annotated = results[0].plot()
            handle_detection(annotated, count)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_streamer(
            key="live",
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
        )

    # ---------- VIDEO UPLOAD ----------
    else:
        video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

        if video:
            with open("input.mp4", "wb") as f:
                f.write(video.read())

            if st.button("Start Detection"):
                cap = cv2.VideoCapture("input.mp4")
                frame_box = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=0.5, device="cpu", verbose=False)
                    boxes = results[0].boxes
                    detected = boxes.cls.cpu().numpy() if boxes else []
                    count = int((detected == POACHER_CLASS_ID).sum())

                    annotated = results[0].plot()
                    handle_detection(annotated, count)

                    frame_box.image(annotated, channels="BGR")

                cap.release()

# ============================
# DATABASE
# ============================
elif menu == "Database" and st.session_state.authenticated:
    st.title("📁 Detection Records")

    data = fetch_detections()
    if not data:
        st.info("No records found")

    for id, ts, count, img_blob in data:
        st.markdown(f"### Detection {id}")
        st.write(f"Time: {ts}")
        st.write(f"Count: {count}")
        img = cv2.imdecode(np.frombuffer(img_blob, np.uint8), cv2.IMREAD_COLOR)
        st.image(img, channels="BGR")
        st.markdown("---")

    if st.button("Clear Database"):
        clear_detections()
        st.success("Database cleared")

# ============================
# HEAT MAP
# ============================
elif menu == "Heat Map" and st.session_state.authenticated:
    st.title("📊 Weekly Poacher Detection Heatmap")

    records = fetch_detections()
    grid = np.zeros((7, 8))

    for _, ts, count, _ in records:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        grid[dt.weekday(), dt.hour // 3] += int(count)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    slots = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21-24"]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect="auto")
    plt.colorbar(im, ax=ax, label="Detections")

    ax.set_xticks(range(8))
    ax.set_yticks(range(7))
    ax.set_xticklabels(slots)
    ax.set_yticklabels(days)

    for i in range(7):
        for j in range(8):
            ax.text(j, i, int(grid[i, j]), ha="center", va="center")

    st.pyplot(fig)
