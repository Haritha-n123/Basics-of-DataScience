import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="Face Detection", layout="wide")
st.title("🔍 Human Face Detection")
st.write("Upload an image to detect if it contains a human face")

# Initialize MediaPipe Face Detection (more accurate than Haar Cascade)
@st.cache_resource
def load_face_detector():
    mp_face_detection = mp.solutions.face_detection
    return mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

face_detector = load_face_detector()
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "webp"])

# Sensitivity slider
st.sidebar.markdown("### Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Lower value = more detections but more false positives"
)

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Detect faces using MediaPipe
    results = face_detector.process(image_bgr)
    
    # Create two columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Detection Result")
        
        # Draw rectangles around detected faces
        image_with_faces = image_np.copy()
        face_count = 0
        
        if results.detections:
            h, w, c = image_np.shape
            for detection in results.detections:
                if detection.score[0] >= confidence_threshold:
                    face_count += 1
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x_min = int(bbox.xmin * w)
                    y_min = int(bbox.ymin * h)
                    x_max = int((bbox.xmin + bbox.width) * w)
                    y_max = int((bbox.ymin + bbox.height) * h)
                    
                    # Draw rectangle
                    cv2.rectangle(image_with_faces, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    # Add confidence score
                    confidence = int(detection.score[0] * 100)
                    cv2.putText(image_with_faces, f'{confidence}%', (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        st.image(image_with_faces, use_column_width=True)
    
    # Display results
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if face_count > 0:
            st.success(f"✅ Face(s) Detected")
        else:
            st.warning("❌ No Face Detected")
    
    with col2:
        st.info(f"Number of faces found: {face_count}")
    
    with col3:
        st.info(f"Image size: {image_np.shape[1]}x{image_np.shape[0]}")
    
    # Detailed information
    if face_count > 0:
        st.markdown("### Face Detection Details")
        if results.detections:
            h, w, c = image_np.shape
            for idx, detection in enumerate(results.detections, 1):
                if detection.score[0] >= confidence_threshold:
                    confidence = int(detection.score[0] * 100)
                    bbox = detection.location_data.relative_bounding_box
                    x_min = int(bbox.xmin * w)
                    y_min = int(bbox.ymin * h)
                    box_width = int(bbox.width * w)
                    box_height = int(bbox.height * h)
                    
                    st.write(f"**Face {idx}:**")
                    st.write(f"- Confidence: {confidence}%")
                    st.write(f"- Position: ({x_min}, {y_min})")
                    st.write(f"- Size: {box_width}×{box_height} pixels")
    else:
        st.markdown("### Result")
        st.write("No human face was detected in the uploaded image at the current confidence threshold.")
        st.write("Try:")
        st.write("- Lowering the confidence threshold in the sidebar")
        st.write("- Ensuring the face is clearly visible")
        st.write("- Checking image quality and lighting")