import numpy as np
import pandas as pd

def preprocess_input(color, brand, price, material_type, brand_means, scaler, color_columns, material_columns):

    # Initialize a dictionary for the new input
    new_data = {}

    # One-hot encode color
    for col in color_columns:
        new_data[col] = [1 if col == f'colour_{color}' else 0]

    # One-hot encode material_type
    for col in material_columns:
        new_data[col] = [1 if col == f'material_type_{material_type}' else 0]

    # Target encode brand using the precomputed brand means
    new_data['brand_encoded'] = [brand_means.get(brand, 0.4)]  # Default to 0 if brand not found

    # Scale price
    new_data['price_scaled'] = scaler.transform([[price]])[0]

    # Convert to DataFrame
    new_df = pd.DataFrame(new_data)

    return new_df