import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client=Groq(
    api_key=api_key
)

# Define function to generate explanation using the LLM (e.g., llama3-8b-8192)
def generate_explanation(user_color, user_brand, user_price, user_material_type, new_product_pred, new_product_pred_proba):
    # Build prediction result text based on whether it's a success or failure
    if new_product_pred == 1:
        prediction_text = f"The model predicts that this product will be a success with {new_product_pred_proba*100:.2f}% confidence."
    else:
        prediction_text = f"The model predicts that this product will fail with {new_product_pred_proba*100:.2f}% confidence."

    # Construct the prompt that will be passed to the LLM
    prompt = f"""
    Based on the following product details:
    - Color: {user_color}
    - Brand: {user_brand}
    - Price: {user_price}
    - Material Type: {user_material_type}

    {prediction_text}

    Please explain why the model predicts this outcome based on the product's features.
    Also suggest ways for improvement
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

