import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your model path

# Define a Video Transformer class for YOLOv8 detection
class VideoTransformer(VideoTransformerBase):
    label_dict = {}  # Class-level dictionary to store labels

    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Perform YOLOv8 object detection
        results = self.model(img)
        boxes = results[0].boxes.xyxy.numpy()  # Get bounding boxes
        labels = results[0].names  # Get class names
        
        # Update the global dictionary with detected labels
        self.update_labels(results)
        
        # Draw bounding boxes and labels on the frame
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            label = labels[int(results[0].boxes.cls[i])]
            print(label)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img
    
    def update_labels(self, results):
        # Get the current detected labels
        current_labels = results[0].names
        boxes = results[0].boxes.xyxy.numpy()
        
        # Update the global dictionary with detected labels
        for i, box in enumerate(boxes):
            label = current_labels[int(results[0].boxes.cls[i])]
            if label in self.label_dict:
                self.label_dict[label] += 1
            else:
                self.label_dict[label] = 1
        
        print(f"Updated Label Dictionary: {self.label_dict}")
    
    def print_label(self):
        st.write("Final Label Dictionary:")
        st.write(self.label_dict)

# Streamlit application
def main():
    st.title("YOLOv8 Object Detection with WebRTC")

    # Create an instance of the VideoTransformer class
    transformer = VideoTransformer(model)

    # WebRTC streamer configuration
    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: transformer,
        media_stream_constraints={"video": True, "audio": True},
    )

    # Button to display the final label_dict
    if st.button("See Final Label Dictionary"):
        VideoTransformer.print_label()

if __name__ == "__main__":
    main()
