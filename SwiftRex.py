import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from deepface import DeepFace
from PIL import ImageTk
from PIL import Image
import os

# Set the OPENCV_OPENCL_DEVICE environment variable to enable Intel acceleration
os.environ['OPENCV_OPENCL_DEVICE'] = 'cpu'  # or 'gpu'


def detect_emotions(file_path):
    # Use the DeepFace library to detect emotions
    result = DeepFace.analyze(file_path, actions=['emotion'])
    # Get the maximum emotion score and corresponding label
    emotion_scores = result[0]['emotion']
    max_score = max(emotion_scores.values())
    max_emotion = [emotion for emotion, score in emotion_scores.items() if score == max_score][0]
    # Return the maximum emotion score and label
    return max_emotion


# Define a function to handle the "Upload Image" button click
def upload_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image using OpenCV
        img = cv2.imread(file_path)
        # Resize the image to fit in the GUI window using standard interpolation
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
        # Convert the image to a PhotoImage object
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img))
        # Set the image in the GUI window
        image_label.config(image=img_tk)
        image_label.image = img_tk
        # Detect emotions from the image and display the result
        max_score = detect_emotions(file_path)
        result_label.config(text=max_score)


# Create a new GUI window
window = tk.Tk()
window.title('Emotion Detection')

# Create a label for the uploaded image
image_label = tk.Label(window)
image_label.pack()

# Create a button to upload an image
upload_button = tk.Button(window, text='Upload Image', command=upload_image)
upload_button.pack()

# Create a label for the detected emotion
result_label = tk.Label(window, font=('Helvetica', 18))
result_label.pack()

# Start the GUI event loop
window.mainloop()
