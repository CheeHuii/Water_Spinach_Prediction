import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 1️⃣ Load the saved model
model_path = 'water_spinach_freshness_model.h5'  # Update with the correct path if needed
model = tf.keras.models.load_model(model_path)
print(f"Loaded model from {model_path}")

# 2️⃣ Load class names (you may hardcode them if not available)
# Example: class_names = ['fresh', 'wilted', 'aging_dying', 'rotten']
# If available from training, replace this with the actual order
class_names = ['fresh', 'wilted', 'aging_dying', 'rotten']  # Update if needed

# 3️⃣ Define image size (same as model input size)
img_size = (224, 224)

def predict_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename()
    if file_path:
        img = image.load_img(file_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Display the image with prediction
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class} ({confidence * 100:.2f}%)")
        plt.axis('off')
        plt.show()

# Run the prediction function
if __name__ == '__main__':
    predict_image()
