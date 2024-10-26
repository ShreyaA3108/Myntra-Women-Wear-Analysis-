import pandas as pd
# Initialize Groq client with API key (replace 'YOUR_API_KEY' with actual key)
df = pd.read_pickle("dataframe_for_success_prob.pkl")

def compute_success_percentage_color(user_color):
    # Filter rows where the colour matches the user_color
    color_data = df[df['colour'] == user_color]
    if color_data.empty:
        return f"No data available for color: {user_color}"
    # Calculate total number of products with the specified color
    total_count = len(color_data)
    # Calculate number of successful products (where target == 1)
    success_count = len(color_data[color_data['target'] == 1])
    # Compute the success percentage
    success_percentage = (success_count / total_count) * 100
    return success_percentage

def compute_success_percentage_material_type(user_material_type):
    # Filter rows where the colour matches the user_color
    material_data = df[df['material_type'] == user_material_type]
    if material_data.empty:
        return f"No data available for color: {user_material_type}"
    # Calculate total number of products with the specified color
    total_count = len(material_data)
    # Calculate number of successful products (where target == 1)
    success_count = len(material_data[material_data['target'] == 1])
    # Compute the success percentage
    success_percentage = (success_count / total_count) * 100
    return success_percentage

def compute_success_percentage_brand(user_brand):
    # Filter rows where the colour matches the user_color
    brand_data = df[df['brand'] == user_brand]
    if brand_data.empty:
        return f"No data available for color: {user_brand}"
    # Calculate total number of products with the specified color
    total_count = len(brand_data)
    # Calculate number of successful products (where target == 1)
    success_count = len(brand_data[brand_data['target'] == 1])
    # Compute the success percentage
    success_percentage = (success_count / total_count) * 100
    return success_percentage
