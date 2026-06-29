import cv2
import streamlit as st
import numpy as np
from datetime import datetime
import os

st.set_page_config(page_title="Motion Detection", layout="wide")

st.title("🎥 Real-Time Motion Detection")

# Sidebar controls
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Motion Threshold", 10, 100, 30)
min_area = st.sidebar.slider("Minimum Contour Area", 5, 100, 20)
save_frames = st.sidebar.checkbox("Save Detected Frames", value=True)
save_dir = "motion_frames"

if save_frames and not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Feed")
    video_placeholder = st.empty()
    
with col2:
    st.subheader("Statistics")
    stats_placeholder = st.empty()

# Start/Stop button
if st.sidebar.button("Start Motion Detection"):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot open webcam. Please check your camera connection.")
    else:
        ret, prev_frame = cap.read()
        if not ret:
            st.error("Cannot read from webcam.")
        else:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            frame_count = 0
            motion_count = 0
            saved_frames = 0
            
            progress_bar = st.progress(0)
            stop_button = st.sidebar.button("Stop Detection")
            
            while cap.isOpened() and not stop_button:
                ret, current_frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and find difference
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, current_gray)
                
                # Apply threshold and find contours
                _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                motion_detected = False
                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue
                    motion_detected = True
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if motion_detected:
                    motion_count += 1
                    # Save frame if motion detected every 10 frames
                    if frame_count % 10 == 0:
                        if save_frames:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(save_dir, f"motion_{timestamp}_{frame_count}.jpg")
                            cv2.imwrite(filename, current_frame)
                            saved_frames += 1
                
                # Display frame (convert BGR to RGB for Streamlit)
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_column_width=True)
                
                # Update statistics
                with stats_placeholder.container():
                    st.metric("Frames Processed", frame_count)
                    st.metric("Motion Detected", motion_count)
                    st.metric("Frames Saved", saved_frames)
                
                # Update for next iteration
                prev_gray = current_gray
                frame_count += 1
                
                # Update progress (cycling indicator)
                progress_bar.progress((frame_count % 100) / 100)
            
            cap.release()
            st.sidebar.success("Motion detection stopped.")

st.sidebar.info(
    "📌 **How to use:**\n"
    "1. Adjust motion detection parameters\n"
    "2. Click 'Start Motion Detection'\n"
    "3. Motion will be highlighted with green boxes\n"
    "4. Frames are saved when motion is detected\n"
    "5. Click 'Stop Detection' to exit"
)