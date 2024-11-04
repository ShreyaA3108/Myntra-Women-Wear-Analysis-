import tkinter as tk
from tkinter import messagebox, Toplevel, filedialog
import pickle
import numpy as np
import pandas as pd
from preprocess_module import preprocess_input
from LLM_module import generate_explanation
from PIL import Image, ImageTk
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
    suggestion_window.geometry("420x400")
    suggestion_window.configure(bg="#23272A")

    # Display the generated explanation text in the window
    tk.Label(
        suggestion_window, text=explanation, wraplength=380, justify="left",
        font=("Arial", 12), fg="white", bg="#23272A",
    ).pack(pady=30)

def predict_value():
    try:
        # Collect input values from the UI
        product_name = product_name_entry.get()  # New input for product name
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
        messagebox.showinfo("Prediction", f"Predicted Outcome for '{product_name}': {result}")

        # Generate explanation using the LLM
        explanation = generate_explanation(color, brand, price, material, new_product_pred, new_product_pred_proba)

        # Enable View Suggestions button and pass explanation
        view_suggestions_button.config(state="normal")
        view_suggestions_button.explanation = explanation

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid inputs.")

def open_webview():
    webview.create_window("Interactive Chart", "combined_interactive_plots_dark_theme.html")
    webview.start()

def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", ".jpg;.jpeg;.png;.gif")])
    if file_path:
        # You can add code to display the image or store the file path as needed
        image_label.config(text=f"Uploaded Image: {file_path.split('/')[-1]}")  # Display the file name

# Create Tkinter Window
root = tk.Tk()
root.title("Product Success Predictor")
root.geometry("600x700")
root.configure(bg="#0F111A")

# Prediction Frame
predict_frame = tk.Frame(root, bg="#0F111A")
predict_frame.pack(expand=True, fill="both", padx=40, pady=40)

# Center elements within predict_frame
predict_frame.columnconfigure(0, weight=1)

# Styling for labels and entry fields
label_style = {
    "font": ("Arial", 15, "bold"), "fg": "#ADBAC7", "bg": "#0F111A",
}
entry_style = {
    "font": ("Arial", 12), "bg": "#1C1F26", "fg": "white", "width": 35,
    "insertbackground": "white", "relief": "flat", "bd": 5,
}
# Button styling with hover effect
def on_enter(e):
    e.widget.config(bg="#3B4F7C")

def on_leave(e):
    e.widget.config(bg="#3A75C4")

# Button styling
button_style = {
    "font": ("Arial", 12, "bold"), 
    "bg": "#3A75C4", 
    "fg": "white", 
    "activebackground": "#2B4D70", 
    "activeforeground": "white", 
    "relief": "raised", 
    "bd": 3, 
    "width": 20
}

# Input fields with consistent spacing
tk.Label(predict_frame, text="Product Name:", **label_style).grid(row=0, column=0, pady=(0, 5))
product_name_entry = tk.Entry(predict_frame, **entry_style)
product_name_entry.grid(row=1, column=0, pady=(0, 20))

tk.Label(predict_frame, text="Color:", **label_style).grid(row=2, column=0, pady=(0, 5))
color_entry = tk.Entry(predict_frame, **entry_style)
color_entry.grid(row=3, column=0, pady=(0, 20))

tk.Label(predict_frame, text="Brand:", **label_style).grid(row=4, column=0, pady=(0, 5))
brand_entry = tk.Entry(predict_frame, **entry_style)
brand_entry.grid(row=5, column=0, pady=(0, 20))

tk.Label(predict_frame, text="Price:", **label_style).grid(row=6, column=0, pady=(0, 5))
price_entry = tk.Entry(predict_frame, **entry_style)
price_entry.grid(row=7, column=0, pady=(0, 20))

tk.Label(predict_frame, text="Material:", **label_style).grid(row=8, column=0, pady=(0, 5))
material_entry = tk.Entry(predict_frame, **entry_style)
material_entry.grid(row=9, column=0, pady=(0, 20))

# Upload Image Button
upload_image_button = tk.Button(
    predict_frame, text="Upload Image", command=upload_image, **button_style
)
upload_image_button.grid(row=10, column=0, pady=(10, 10))

# Label to show uploaded image name
image_label = tk.Label(predict_frame, text="", **label_style)
image_label.grid(row=11, column=0, pady=(10, 20))


# Buttons for Prediction and Navigation with consistent spacing
predict_button = tk.Button(predict_frame, text="Predict", command=predict_value, **button_style)
predict_button.grid(row=12, column=0, pady=(10, 10))
predict_button.bind("<Enter>", on_enter)
predict_button.bind("<Leave>", on_leave)

view_suggestions_button = tk.Button(
    predict_frame, text="View Suggestions", state="disabled", 
    command=lambda: display_suggestions(view_suggestions_button.explanation),
    **button_style
)
view_suggestions_button.grid(row=13, column=0, pady=(10, 10))
view_suggestions_button.bind("<Enter>", on_enter)
view_suggestions_button.bind("<Leave>", on_leave)

visualize_button = tk.Button(
    predict_frame, text="Show Visualizations", command=open_webview, **button_style
)
visualize_button.grid(row=14, column=0, pady=(10, 10))
visualize_button.bind("<Enter>", on_enter)
visualize_button.bind("<Leave>", on_leave)

# Start with the Prediction Frame
show_frame(predict_frame)

# Run the Tkinter event loop
root.mainloop()
