import torch
import torchvision
import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import matplotlib.image as mpimg
import random
from glob import glob


camera_indices = []
for i in range(10):  # Check up to 10 camera indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        camera_indices.append(i)
        cap.release()
    else:
        break

if not camera_indices:
    print("No cameras found.")
else:
    print(f"Available camera indices: {camera_indices}")


model=YOLO('Bestmodel4thrun.pt')  # Load the trained YOLOv11 model
if not camera_indices:
    print("No camera found. Please run the camera detection cell above.")
else:
    # Use the first available camera
    cap = cv2.VideoCapture(camera_indices[0])
    
    if not cap.isOpened():
        print(f"Error: Could not open video stream from camera index {camera_indices[0]}")
    else:
        print("Press 'q' to quit the live feed.")
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run YOLOv8 inference on the frame
            results = model(frame, verbose=False)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the resulting frame
            cv2.imshow('YOLOv8 Live Detection', annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    # Add a small delay to ensure windows are closed in Jupyter
    for i in range(5):
        cv2.waitKey(1)