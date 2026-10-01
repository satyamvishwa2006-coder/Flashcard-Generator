from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get Hugging Face API token
token = os.getenv("HF_TOKEN")

# Create Hugging Face client
client = InferenceClient(
    api_key=token
)

# Ask user for topic
topic = input("Enter the topic for flashcards: ")

# Generate flashcards
response = client.chat_completion(
    model="zai-org/GLM-5.3",
    messages=[
        {
            "role": "user",
            "content": f"""
Create 5 educational flashcards about {topic}.

For every flashcard use this format:

Q: Question
A: Answer

Keep the questions simple and useful for students.
"""
        }
    ],
    max_tokens=800
)

# Display flashcards
print("\n===== FLASHCARDS =====\n")
print(response.choices[0].message.content)