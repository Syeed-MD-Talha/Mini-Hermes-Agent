"""
Simple Q&A script using the Fireworks AI API.

Reads FIREWORKS_API_KEY and FIREWORKS_MODEL from a .env file,
then answers whatever question the user types.

Setup:
    pip install openai python-dotenv

.env file (same folder as this script) should contain:
    FIREWORKS_API_KEY=fw_xxxxxxxx
    FIREWORKS_MODEL=accounts/fireworks/models/llama-v3p1-70b-instruct

Run:
    python main.py
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI


def load_config():
    """Load API key and model name from .env file."""
    load_dotenv()  # reads .env from current directory

    api_key = os.getenv("FIREWORKS_API_KEY")
    model = os.getenv("FIREWORKS_MODEL")

    if not api_key:
        print("Error: FIREWORKS_API_KEY not found in .env file.")
        sys.exit(1)
    if not model:
        print("Error: FIREWORKS_MODEL not found in .env file.")
        sys.exit(1)

    return api_key, model


def ask_question(client: OpenAI, model: str, question: str) -> str:
    """Send the question to the model and return the text answer."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": question}
        ],
    )

    return response.choices[0].message.content


def main():
    api_key, model = load_config()
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    print(f"Using Fireworks model: {model}")
    print("Type your question (or 'exit' to quit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            answer = ask_question(client, model, question)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()