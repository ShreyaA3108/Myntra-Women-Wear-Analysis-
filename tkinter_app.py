import tkinter as tk
from tkinter import messagebox, Toplevel
import pickle
import numpy as np
import pandas as pd
from preprocess_module import preprocess_input
from LLM_module import generate_explanation  # Import the function to generate explanations
from PIL import Image, ImageTk  # Ensure you have Pillow installed
import webview

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

def display_suggestions(explanation):
    suggestion_window = Toplevel(root)
    suggestion_window.title("Product Success Suggestions")
    suggestion_window.geometry("400x400")
    suggestion_window.configure(bg="#2B2B2B")  # Dark background

    # Display the generated explanation text in the window
    tk.Label(
        suggestion_window, text=explanation, wraplength=350, justify="left",
        font=("Arial", 12), fg="white", bg="#2B2B2B"
    ).pack(pady=20)

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
        new_product_pred = voting_clf.predict(new_product_df)[0]
        new_product_pred_proba = voting_clf.predict_proba(new_product_df)[0, 1]

        # Display the prediction
        result = "Success" if new_product_pred == 1 else "Loss"
        messagebox.showinfo("Prediction", f"Predicted Outcome: {result}")

        # Generate explanation using the LLM
        explanation = generate_explanation(color, brand, price, material, new_product_pred, new_product_pred_proba)

        # Enable View Suggestions button and pass explanation
        view_suggestions_button.config(state="normal")
        view_suggestions_button.explanation = explanation  # Save explanation for viewing

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid inputs.")

def open_webview():
    # Open a new webview window with the HTML chart
    webview.create_window("Interactive Chart", "combined_interactive_plots.html")
    webview.start()  # Starts the webview event loop

# Create Tkinter Window
root = tk.Tk()
root.title("Product Success Predictor")
root.geometry("600x600")
root.configure(bg="#1E1E1E")  # Black background

# -------- Prediction Frame --------
predict_frame = tk.Frame(root, bg="#1E1E1E")  # Dark frame background
predict_frame.pack(expand=True, fill="both")

# Center elements within predict_frame
predict_frame.columnconfigure(0, weight=1)
predict_frame.rowconfigure(list(range(10)), weight=1)  # Adds space around elements

# Styling for labels and entry fields
label_style = {"font": ("Arial", 14), "fg": "white", "bg": "#1E1E1E"}
entry_style = {"font": ("Arial", 12), "bg": "#2B2B2B", "fg": "white", "width": 30}

# Input fields
tk.Label(predict_frame, text="Color:", **label_style).grid(row=1, column=0, sticky="n", pady=5)
color_entry = tk.Entry(predict_frame, **entry_style)
color_entry.grid(row=2, column=0, pady=5)

tk.Label(predict_frame, text="Brand:", **label_style).grid(row=3, column=0, sticky="n", pady=5)
brand_entry = tk.Entry(predict_frame, **entry_style)
brand_entry.grid(row=4, column=0, pady=5)

tk.Label(predict_frame, text="Price:", **label_style).grid(row=5, column=0, sticky="n", pady=5)
price_entry = tk.Entry(predict_frame, **entry_style)
price_entry.grid(row=6, column=0, pady=5)

tk.Label(predict_frame, text="Material:", **label_style).grid(row=7, column=0, sticky="n", pady=5)
material_entry = tk.Entry(predict_frame, **entry_style)
material_entry.grid(row=8, column=0, pady=5)

# Button styling
button_style = {
    "font": ("Arial", 12, "bold"), 
    "bg": "#3A75C4", 
    "fg": "white", 
    "activebackground": "#2B4D70", 
    "activeforeground": "white", 
    "relief": "raised", 
    "bd": 3
}

# Buttons for Prediction and Navigation
predict_button = tk.Button(predict_frame, text="Predict", command=predict_value, **button_style)
predict_button.grid(row=9, column=0, pady=10)

# View Suggestions button (initially disabled)
view_suggestions_button = tk.Button(
    predict_frame, text="View Suggestions", state="disabled", 
    command=lambda: display_suggestions(view_suggestions_button.explanation),
    **button_style
)
view_suggestions_button.grid(row=10, column=0, pady=10)

# Button to open the interactive chart directly
visualize_button = tk.Button(
    predict_frame, text="Show Visualizations", command=open_webview, **button_style
)
visualize_button.grid(row=11, column=0, pady=10)

# Start with the Prediction Frame
show_frame(predict_frame)

# Run the Tkinter event loop
root.mainloop()



