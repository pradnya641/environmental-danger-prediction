
import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

image_path = Path(__file__).parent / "test_image.jpg"

with open(image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

client = genai.Client(
    api_key=api_key,
    http_options={"timeout": 120000}
)

models_to_test = [
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite"
]

for model_name in models_to_test:

    print(f"\nTesting model: {model_name}")

    try:
        response = client.interactions.create(
            model=model_name,
            input=[
                {
                    "type": "text",
                    "text": "Describe this image in exactly three sentences."
                },
                {
                    "type": "image",
                    "data": image_data,
                    "mime_type": "image/jpeg"
                }
            ]
        )

        print("\nSUCCESS!")
        print(response.output_text)

    except Exception as error:
        print("\nFAILED")
        print(type(error).__name__, error)
