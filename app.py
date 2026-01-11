import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
from twilio.rest import Client

# Database imports
from database import initialize_database, save_detection, fetch_detections, clear_detections, add_user, verify_user


# ============================
# Twilio Configuration
# ============================
account_sid = 'AC0e02890ff0e74589d760838a92b1d1b7'
auth_token = '3f51606d55ed606e7850558cbbba4437'
twilio_number = '+18444825976'
your_phone_number = '+916363656664'

client = Client(account_sid, auth_token)


def send_sms_alert(message):
    try:
        msg = client.messages.create(body=message, from_=twilio_number, to=your_phone_number)
        return msg.sid
    except Exception as e:
        st.error(f"Failed to send SMS: {e}")
        return None


# ============================
# Load YOLO Model
# ============================
model_path = r"runs\detect\train11\weights\best.pt"

if not os.path.exists(model_path):
    st.error(f"YOLO model not found at {model_path}")
else:
    model = YOLO(model_path)

# Initialize DB
initialize_database()


# ============================
# Streamlit App Setup
# ============================
st.set_page_config(page_title="Anti-Poaching System", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

st.sidebar.title("Navigation")

menu = st.sidebar.selectbox(
    "Menu",
    ["Login", "Sign Up"] if not st.session_state["authenticated"]
    else ["Home", "Dashboard", "Database", "Hash Map", "Logout"]
)


# ============================
# LOGIN
# ============================
if menu == "Login" and not st.session_state["authenticated"]:
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(username, password):
            st.success("Login Successful!")
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
        else:
            st.error("Invalid credentials")


# ============================
# SIGN UP
# ============================
elif menu == "Sign Up" and not st.session_state["authenticated"]:
    st.subheader("📝 Create New Account")

    new_user = st.text_input("New Username")
    new_pass = st.text_input("Password", type="password")
    conf_pass = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_pass == conf_pass:
            add_user(new_user, new_pass)
            st.success("Account created successfully! Go to Login.")
        else:
            st.error("Passwords do not match!")


# ============================
# LOGOUT
# ============================
elif menu == "Logout":
    st.session_state["authenticated"] = False
    st.success("Logged out successfully!")
    st.stop()


# ============================
# HOME PAGE
# ============================
elif menu == "Home" and st.session_state["authenticated"]:
    st.title("🦌 Anti-Poaching Detection System")
    st.markdown("""
    This system uses **YOLO-based deep learning** to detect poachers in real time.

    **Features:**
    - Live Camera Detection  
    - Video File Detection  
    - Automatic SMS Alerts  
    - Detection Storage in Database  
    - Weekly Heat Map (Hash Map)  
    """)


# ============================
# DETECTION PAGE
# ============================
elif menu == "Dashboard" and st.session_state["authenticated"]:
    st.title("🎥 Poacher Detection Dashboard")

    mode = st.radio("Select Mode", ["Live Camera", "Upload Video"])

    def detect_and_display(input_source):
        cap = cv2.VideoCapture(input_source)
        stframe = st.empty()
        sms_log = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detected_classes = results[0].boxes.cls.cpu().numpy()
            class_0 = (detected_classes == 0).sum()

            annotated = results[0].plot()

            # Save and send alerts
            if class_0 > 0:
                t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                sid = send_sms_alert(f"🚨 Poacher detected at {t} — Count: {class_0}")
                sms_log.write(f"SMS Alert Sent! SID: {sid}")

                _, img_encoded = cv2.imencode(".jpg", annotated)
                save_detection(t, int(class_0), img_encoded.tobytes())

            stframe.image(annotated, channels="BGR")

        cap.release()

    if mode == "Upload Video":
        video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if video:
            with open("uploaded_video.mp4", "wb") as f:
                f.write(video.read())

            if st.button("Start Detection"):
                detect_and_display("uploaded_video.mp4")

    else:
        if st.button("Start Live Camera Detection"):
            detect_and_display(0)


# ============================
# DATABASE PAGE
# ============================
elif menu == "Database" and st.session_state["authenticated"]:
    st.title("📁 Poacher Detection Records")

    detections = fetch_detections()

    if len(detections) == 0:
        st.info("No detections recorded yet.")
    else:
        for id, ts, count, img_blob in detections:
            st.markdown(f"### Detection ID: {id}")
            st.write(f"**Time:** {ts}")
            st.write(f"**Count:** {count}")

            img = cv2.imdecode(np.frombuffer(img_blob, np.uint8), cv2.IMREAD_COLOR)
            st.image(img, channels="BGR")

            st.markdown("---")

    if st.button("Clear All Data"):
        clear_detections()
        st.success("All records cleared!")


# ============================
# HASH MAP PAGE (WITHOUT SEABORN)
# ============================
elif menu == "Hash Map" and st.session_state["authenticated"]:
    st.title("📊 Weekly Poacher Detection Heatmap (7×8 Grid)")

    raw = fetch_detections()
    grid = np.zeros((7, 8))

    for _, ts, count, _ in raw:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        day = dt.weekday()
        interval = dt.hour // 3

        try:
            count = int(count)
        except:
            count = 0

        grid[day, interval] += count

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    intervals = ['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21', '21-24']

    fig, ax = plt.subplots(figsize=(10, 4))

    heatmap = ax.imshow(grid, aspect='auto')

    cbar = plt.colorbar(heatmap)
    cbar.set_label("Detection Count", rotation=270, labelpad=15)

    ax.set_xticks(np.arange(len(intervals)))
    ax.set_yticks(np.arange(len(days)))
    ax.set_xticklabels(intervals)
    ax.set_yticklabels(days)

    for i in range(len(days)):
        for j in range(len(intervals)):
            ax.text(j, i, int(grid[i, j]), ha='center', va='center', color='black')

    ax.set_title("Weekly Poacher Detection Heatmap (Matplotlib Only)")

    st.pyplot(fig)
