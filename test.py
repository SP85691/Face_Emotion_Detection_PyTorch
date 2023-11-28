import random
import os
import cv2
import torch
import nbimporter
from trainmodel import SimpleCNN
import matplotlib.pyplot as plt

model = SimpleCNN()

# Load the saved model
model.load_state_dict(torch.load('emotiondetector.h5'))

# Put the model in evaluation mode (important if you're going to use it for inference)
model.eval()

test_dir = 'dataset/images/images/test'
random_label = random.choice(os.listdir(test_dir))
random_image_name = random.choice(os.listdir(os.path.join(test_dir, random_label)))
random_image_path = os.path.join(test_dir, random_label, random_image_name)

# random_image_path = 'dataset/images/test/454.jpg'

random_image = cv2.imread(random_image_path, cv2.IMREAD_GRAYSCALE)
random_image = cv2.resize(random_image, (48, 48))
random_image = random_image / 255.0
random_image = torch.Tensor(random_image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Pass the image through the model
with torch.no_grad():
    predicted_emotion = model(random_image)

# Get the predicted label
predicted_label = label[predicted_emotion.argmax()]

# Save the image as a file
output_path = 'predicted_image.png'
cv2.imwrite(output_path, cv2.imread(random_image_path))

print(f"The predicted emotion for the random image is: {predicted_label}")
print(f"Image saved at: {output_path}")
