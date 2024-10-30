import os
from dotenv import load_dotenv
from compute_success import compute_success_percentage_color,compute_success_percentage_material_type,compute_success_percentage_brand
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client=Groq(
    api_key=api_key
)

# Define function to generate explanation using the LLM (e.g., llama3-8b-8192)
def generate_explanation(user_color, user_brand, user_price, user_material_type, new_product_pred, new_product_pred_proba):
    brand_success=compute_success_percentage_brand(user_brand)
    material_type_success=compute_success_percentage_material_type(user_material_type)
    color_success=compute_success_percentage_color(user_color)
    # Build prediction result text based on whether it's a success or failure
    if new_product_pred == 1:
        prediction_text = f"The product will be a success with {new_product_pred_proba*100:.2f}% confidence."
    else:
        prediction_text = f"The product will fail with {new_product_pred_proba*100:.2f}% confidence."

    # Construct the prompt that will be passed to the LLM
    prompt = f"""
    Based on the following product details:
    - Color: {user_color} with {color_success}% success of this color
    - Brand: {user_brand} with {brand_success}% success of this brand
    - Price: {user_price} 
    - Material Type: {user_material_type} with {material_type_success}% success of this material_type

    {prediction_text}
    Based on this , please provide a generalised view to why this is the outcome.
    Also suggest ways the company can improve is product.please provide everything in short.please dont say lack of dataset only provide your generalised view based on the given success probabilities.
    please answer in short only.
    """

    # Call the Groq LLM (Llama3-8b-8192) to generate the explanation
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama3-8b-8192",  # Using Groq's LLM model
    )

    # Extract and return the generated explanation
    explanation = chat_completion.choices[0].message.content
    return explanation

