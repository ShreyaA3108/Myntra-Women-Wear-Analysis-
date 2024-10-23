import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
import pandas as pd
from preprocess_module import preprocess_input
from tkinter import Toplevel
from PIL import Image, ImageTk  # Ensure you have `Pillow` installed

# Load the pickled components
with open('model_components.pkl', 'rb') as f:
    components = pickle.load(f)

# Load the Pickle Model (Voting Classifier)
with open('model.pkl', 'rb') as f:
    voting_clf = pickle.load(f)

# Unpack the components
brand_means = components['brand_means']
scaler = components['scaler']
color_columns = components['color_columns']
material_columns = components['material_columns']

# Function to switch between frames
def show_frame(frame):
    frame.tkraise()


def predict_value():
    try:
        # Collect input values from the UI
        color = color_entry.get()
        brand = brand_entry.get()
        price = float(price_entry.get())
        material = material_entry.get()

        # Preprocess the inputs
        new_product_df = preprocess_input(
            color, brand, price, material,
            brand_means, scaler, color_columns, material_columns
        )

        # Predict the outcome using the voting classifier
        new_product_pred = voting_clf.predict(new_product_df)
        new_product_pred_proba = voting_clf.predict_proba(new_product_df)[:, 1]

        # Display the prediction
        messagebox.showinfo("Prediction", f"Predicted Outcome: {'Success' if new_product_pred[0] == 1 else 'Loss'}")

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid inputs.")



# Create Tkinter Window
root = tk.Tk()
root.title("Product Success Predictor")
root.geometry("600x600")

# Create Frames
predict_frame = tk.Frame(root)
visualization_frame = tk.Frame(root)

for frame in (predict_frame, visualization_frame):
    frame.grid(row=0, column=0, sticky='nsew')

# -------- Prediction Frame --------
tk.Label(predict_frame, text="Color:").pack(pady=5)
color_entry = tk.Entry(predict_frame)
color_entry.pack(pady=5)

tk.Label(predict_frame, text="Brand:").pack(pady=5)
brand_entry = tk.Entry(predict_frame)
brand_entry.pack(pady=5)

tk.Label(predict_frame, text="Price:").pack(pady=5)
price_entry = tk.Entry(predict_frame)
price_entry.pack(pady=5)

tk.Label(predict_frame, text="Material:").pack(pady=5)
material_entry = tk.Entry(predict_frame)
material_entry.pack(pady=5)

# Buttons for Prediction and Navigation
predict_button = tk.Button(predict_frame, text="Predict", command=predict_value)
predict_button.pack(pady=10)

visualize_button = tk.Button(predict_frame, text="Show Visualizations", 
                             command=lambda: show_frame(visualization_frame))
visualize_button.pack(pady=10)

# -------- Visualization Frame --------
# Load and display the image
img = Image.open("average_price_rating.png")  # Use your image path
photo = ImageTk.PhotoImage(img)

img_label = tk.Label(visualization_frame, image=photo)
img_label.image = photo  # Keep a reference to avoid garbage collection
img_label.pack(padx=10, pady=10)

# Button to go back to Prediction Frame
back_button = tk.Button(visualization_frame, text="Back to Prediction", 
                        command=lambda: show_frame(predict_frame))
back_button.pack(pady=10)

# Start with the Prediction Frame
show_frame(predict_frame)

# Run the Tkinter event loop
root.mainloop()