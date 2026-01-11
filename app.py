import streamlit as st
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO

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
# Streamlit Config (FIRST)
# ============================
st.set_page_config(page_title="Anti-Poaching System", layout="wide")

# ============================
# Twilio Configuration (SAFE)
# ============================
SMS_ENABLED = True
try:
    from twilio.rest import Client

    account_sid = st.secrets["TWILIO_SID"]
    auth_token = st.secrets["TWILIO_TOKEN"]
    twilio_number = st.secrets["TWILIO_FROM"]
    your_phone_number = st.secrets["TWILIO_TO"]

    client = Client(account_sid, auth_token)

except Exception:
    SMS_ENABLED = False
    st.warning("⚠️ SMS alerts disabled (Twilio secrets not configured)")


def send_sms_alert(message):
    if not SMS_ENABLED:
        return None
    try:
        msg = client.messages.create(
            body=message,
            from_=twilio_number,
            to=your_phone_number
        )
        return msg.sid
    except Exception as e:
        st.error(f"SMS failed: {e}")
        return None


# ============================
# Load YOLO Model
# ============================
MODEL_PATH = "runs/detect/train11/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"YOLO model not found at {MODEL_PATH}")
    st.stop()

model = YOLO(MODEL_PATH)

# ============================
# Initialize Database
# ============================
initialize_database()

# ============================
# Session State
# ============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ============================
# Sidebar
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

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state.authenticated = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

# ============================
# SIGN UP
# ============================
elif menu == "Sign Up" and not st.session_state.authenticated:
    st.subheader("📝 Create Account")

    new_user = st.text_input("New Username")
    new_pass = st.text_input("Password", type="password")
    conf_pass = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_pass == conf_pass:
            add_user(new_user, new_pass)
            st.success("Account created. Please login.")
        else:
            st.error("Passwords do not match")

# ============================
# LOGOUT
# ============================
elif menu == "Logout":
    st.session_state.authenticated = False
    st.success("Logged out")
    st.stop()

# ============================
# HOME
# ============================
elif menu == "Home" and st.session_state.authenticated:
    st.title("🦌 Anti-Poaching Detection System")
    st.markdown("""
    **YOLO-based AI system to detect poachers from video footage**

    **Features**
    - Video-based detection
    - Automatic SMS alerts
    - Detection history database
    - Weekly heat map analytics
    """)

# ============================
# DASHBOARD (VIDEO ONLY)
# ============================
elif menu == "Dashboard" and st.session_state.authenticated:
    st.title("🎥 Poacher Detection (Upload Video)")

    video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"]
    )

    if video:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video.read())

        if st.button("Start Detection"):
            cap = cv2.VideoCapture("uploaded_video.mp4")
            frame_box = st.empty()
            log_box = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated = results[0].plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                classes = results[0].boxes.cls.cpu().numpy()
                count = int((classes == 0).sum())

                if count > 0:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    sid = send_sms_alert(
                        f"🚨 Poacher detected at {timestamp} | Count: {count}"
                    )

                    log_box.write(f"Detection logged | SMS SID: {sid}")

                    _, buf = cv2.imencode(".jpg", annotated)
                    save_detection(timestamp, count, buf.tobytes())

                frame_box.image(annotated)

            cap.release()
            st.success("Detection completed")

# ============================
# DATABASE PAGE
# ============================
elif menu == "Database" and st.session_state.authenticated:
    st.title("📁 Detection Records")

    records = fetch_detections()

    if not records:
        st.info("No detections yet")
    else:
        for rec_id, ts, count, img_blob in records:
            st.markdown(f"### Detection ID: {rec_id}")
            st.write(f"Time: {ts}")
            st.write(f"Count: {count}")

            img = cv2.imdecode(
                np.frombuffer(img_blob, np.uint8),
                cv2.IMREAD_COLOR
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img)

            st.markdown("---")

    if st.button("Clear All Records"):
        clear_detections()
        st.success("Database cleared")

# ============================
# HEAT MAP
# ============================
elif menu == "Heat Map" and st.session_state.authenticated:
    st.title("📊 Weekly Poacher Detection Heat Map")

    data = fetch_detections()
    grid = np.zeros((7, 8))

    for _, ts, count, _ in data:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        grid[dt.weekday(), dt.hour // 3] += int(count)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    intervals = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21-24"]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect="auto")

    plt.colorbar(im, ax=ax, label="Detection Count")

    ax.set_xticks(range(8))
    ax.set_yticks(range(7))
    ax.set_xticklabels(intervals)
    ax.set_yticklabels(days)

    for i in range(7):
        for j in range(8):
            ax.text(j, i, int(grid[i, j]), ha="center", va="center")

    ax.set_title("Weekly Detection Heat Map")

    st.pyplot(fig)
