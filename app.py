import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import pandas as pd
st.set_page_config(
    page_title="GuardianAI - Real-time Security",
    page_icon="🛡️",
    layout="wide"
)
# --- AI Core Logic ---
# This section contains the functions that Person 1, the AI Engineer, created.
# You will call these functions from the Streamlit interface.

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt") 

# Define the classes we are interested in
# These are standard COCO dataset class names.
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# A simple function to draw detections on a frame
def draw_detections(frame, boxes, class_ids):
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES[int(class_id)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put class name and confidence
        cv2.putText(frame, class_name, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Function to process a single frame (for live feed)
def process_frame(frame):
    """
    Processes a single video frame with the YOLO model.
    Returns the annotated frame and a list of detected class names.
    """
    results = model(frame, verbose=False)
    
    # Check if results are not empty
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        
        annotated_frame = draw_detections(frame.copy(), boxes, class_ids)
        
        # Get unique detected class names
        detected_classes = [CLASS_NAMES[int(cid)] for cid in class_ids]
        detections = list(set(detected_classes)) # Use set to get unique names
    else:
        annotated_frame = frame
        detections = []

    return annotated_frame, detections

# Function to process an entire video file
def process_video_file(input_path, output_path):
    """
    Processes a video file, draws detections, and saves the output.
    Returns a list of dictionaries with timestamps and detected events.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return []

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    alerts = []
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame using the same logic as the live feed
        annotated_frame, detections = process_frame(frame)
        
        # Log alerts
        if detections:
            timestamp = frame_number / fps
            alerts.append({
                "Timestamp (s)": f"{timestamp:.2f}",
                "Event": ", ".join(detections)
            })

        out.write(annotated_frame)
        frame_number += 1

    cap.release()
    out.release()
    
    # Remove duplicates from alerts by converting to a set of tuples and back
    if alerts:
        unique_alerts = list({(alert['Timestamp (s)'], alert['Event']) for alert in alerts})
        alerts = [{"Timestamp (s)": ts, "Event": ev} for ts, ev in sorted(unique_alerts)]
        # This is a simple way to consolidate; more advanced logic could be used
        # For now, let's create a DataFrame and drop duplicates on the "Event" column
        df = pd.DataFrame(alerts)
        df_unique_events = df.drop_duplicates(subset=['Event'], keep='first')
        return df_unique_events
        
    return []


# --- Streamlit Web App Interface ---

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Home", "Analyze Video File", "Live Feed Analysis"]
)

if app_mode == "Home":
    st.title("GuardianAI: Your AI-Powered Security Assistant")
    st.markdown("""
    Welcome to GuardianAI! This application uses state-of-the-art computer vision to monitor and analyze video footage for specific events.

    **Choose an option from the sidebar to get started:**

    - **Analyze Video File:** Upload a pre-recorded video to detect events.
    - **Live Feed Analysis:** Use your webcam for real-time event detection.

    This project demonstrates the power of the YOLOv8 model for object detection in a real-world security context.
    """)

elif app_mode == "Analyze Video File":
    st.header("Analyze a Pre-recorded Video")

    uploaded_file = st.file_uploader("Upload a video file (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name

        st.video(temp_video_path)

        if st.button("Analyze Video"):
            with st.spinner("Analyzing... This may take a few moments."):
                output_video_path = "output_video.mp4"

                # This is where you call Person 1's function
                alerts = process_video_file(temp_video_path, output_video_path)

            st.success("Analysis Complete!")
            st.video(output_video_path)

            if not alerts.empty:
                st.header("Detected Events:")
                st.table(alerts) # Display alerts in a nice table
            else:
                st.info("No specific events were detected in this video.")

elif app_mode == "Live Feed Analysis":
    st.header("Analyze Live Webcam Feed")
    st.warning("This feature will request access to your webcam. Ensure you have given your browser permission.")

    # Use session state to manage the run state of the webcam
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False

    if st.button('Start Live Feed'):
        st.session_state.run_webcam = True

    if st.button('Stop Live Feed'):
        st.session_state.run_webcam = False
        st.info("Live feed stopped.")

    if st.session_state.run_webcam:
        st.info("Webcam feed is running... (Click 'Stop Live Feed' to end)")

        # Placeholder for the video frames
        frame_placeholder = st.empty()

        # Open the webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            while cap.isOpened() and st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.write("The video stream has ended.")
                    break

                # This is where you call Person 1's new function
                annotated_frame, detections = process_frame(frame)

                # Display the processed frame
                frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()