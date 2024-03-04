# Importing necessary Libraries and Modules

import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import *
from keras.preprocessing import image

# Creating Window for Opening Image
def open_image():
    file_path = filedialog.askopenfilename(title="Open Image File")
    if file_path:
        display_image(file_path)
        detect_covid(file_path)

# Displaying the image
def display_image(file_path):
    image = Image.open(file_path)
    resizedImg = image.resize((600,600))
    photo = ImageTk.PhotoImage(resizedImg)
    image_label.config(image=photo)
    image_label.photo = photo
    status_label.config(text = f"Image loaded: {file_path}")

# Detecting Covid-19
def detect_covid(file_path):
    # Importing the Model
    Output.delete("1.0", END)
    model = load_model('model_adv2_tuberculosis.h5')

    # The location for the image
    path = file_path

    img = image.load_img(path, target_size=(224, 224))

    # preparing the image
    img = image.img_to_array(img) / 255
    img = np.array([img])

    # Predicting the image
    predictions = (model.predict(img) > 0.5).astype("int32")

    if predictions == 1:
        print("Tuberculosis")
        Output.insert(END, "Tuberculosis")
        Output.tag_add("center", 1.0, "end")
    else:
        print("Covid")
        Output.insert(END, "Covid-19")
        Output.tag_add("center", 1.0, "end")

# Creating the GUI window
root = tk.Tk()
ico = Image.open('SARS-CoV-2_without_background.png')
photo = ImageTk.PhotoImage(ico)
root.wm_iconphoto(False, photo)
root.geometry("1920x1080")
root.title("Covid-19 Detector")
text_widget = tk.Text(root, wrap=tk.WORD, height=10, width=35)
open_button = tk.Button(root, text="Open Image", command=open_image)
exit_button = tk.Button(root, text="Exit", command=root.destroy)
Output = Text(root, height=1, width=20, bg='cyan')
Output.tag_configure("center", justify="center")
font_touple = ("Times New Roman", 20 ,"bold")
Output.configure(font=font_touple)
status_label = tk.Label(root, text="", padx=20, pady=10)
Output.pack()
open_button.pack(padx=20, pady=10)
exit_button.pack(padx=20)
image_label = tk.Label(root, height=600, width=600, borderwidth=2, relief='solid')
image_label.pack(padx=20, pady=20)
status_label.pack()
root.mainloop()
