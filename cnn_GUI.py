import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# Load pre-trained VGG16 model + higher level layers
model = VGG16(weights='imagenet', include_top=True)

def process_img_from_url(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img

def activation_maps(model, img_array, layer_names):
    activations = []
    for layer_name in layer_names:
        intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        activations.append(intermediate_model.predict(img_array))
    return activations

def display_activation_maps(activations, layer_names):
    for layer_name, activation in zip(layer_names, activations):
        num_filters = activation.shape[-1]
        plt.figure(figsize=(16, 8))
        plt.suptitle(layer_name)

        for i in range(min(num_filters, 8)):
            plt.subplot(1, 8, i + 1)
            plt.imshow(activation[0, :, :, i], cmap='gray')
            plt.axis('off')

        plt.show()

def on_upload():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        global img_array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

def on_detect():
    if img_array is not None:
        layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1']
        activations = activation_maps(model, img_array, layer_names)
        display_activation_maps(activations, layer_names)

# Create GUI window
root = tk.Tk()
root.title("Task 1")
root.configure(bg='#2E2E2E')  # Dark background color

frame = tk.Frame(root, bg='#2E2E2E')
frame.pack(padx=100, pady=100)

title = tk.Label(frame, text="CNN Filters for Age Detection", font=("Arial", 16), fg='#FFFFFF', bg='#2E2E2E')
title.pack(side="top", padx=10, pady=10)

panel = tk.Label(frame, bg='#2E2E2E')
panel.pack(side="top", padx=10, pady=10)

btn_upload = tk.Button(frame, text="Upload Image", command=on_upload, bg='#3E3E3E', fg='#FFFFFF')
btn_upload.pack(side="left", padx=10, pady=10)

btn_detect = tk.Button(frame, text="Detect", command=on_detect, bg='#3E3E3E', fg='#FFFFFF')
btn_detect.pack(side="right", padx=10, pady=10)

img_array = None  # Initialize img_array variable

root.mainloop()
