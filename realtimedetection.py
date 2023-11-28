import cv2
import torch
import numpy as np
import nbimporter
from trainmodel import SimpleCNN

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load('emotiondetector.h5'))
model.eval()

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

webcam = 0
vid_path = "./dataset/test_case_1.mp4"

# Initialize the webcam89
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = torch.Tensor(face).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Pass the face through the model
        with torch.no_grad():
            predicted_emotion = model(face)

        # Get the predicted label
        predicted_label = label[predicted_emotion.argmax()]

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
