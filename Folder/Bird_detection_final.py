import cv2
import torch
import pygame
import time
import os

# Initialize pygame for sound
pygame.mixer.init()
alert_sound_path = "C:\\Users\\admin\\Desktop\\Bird_detection - Copy\\alert1.wav"
if os.path.exists(alert_sound_path):
    alert_sound = pygame.mixer.Sound(alert_sound_path)
else:
    print("Error: Alert sound file not found.")
    exit()

# Load YOLOv5 pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Variables to manage detection state
bird_detected = False
last_alert_time = 0
alert_cooldown = 1  # seconds to wait before allowing another alert

try:
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Resize frame for performance
        img = cv2.resize(img, (640, 480))

        # Perform object detection on the frame
        results = model(img)

        # Check if any bird is detected
        detected = False
        for obj in results.xyxy[0]:  # Iterate through detected objects
            class_id = int(obj[5])  # Class ID
            label = results.names[class_id]
            if "bird" in label.lower():  # Check if the detected object is a bird
                detected = True
                break

        # Handle detection logic
        current_time = time.time()
        if detected:
            if not bird_detected and current_time - last_alert_time > alert_cooldown:
                bird_detected = True
                last_alert_time = current_time
                print("Bird detected!")
                alert_sound.play()
        else:
            bird_detected = False

        # Render the results on the frame
        results.render()  # Draw detections on the image
        cv2.imshow('Real-Time Bird Detection', img)

        # Exit loop on pressing 'Esc'
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Escape key
            break
finally:
    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
