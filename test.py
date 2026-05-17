#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv

def fetch_groq_models():
    """
    Fetches the list of available models from Groq using the API key in .env.
    Returns a list of model IDs (strings).
    """

    # Load .env file
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment or .env file.")

    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()

    # Groq returns: { "data": [ { "id": "...", ... }, ... ] }
    models = [m["id"] for m in data.get("data", [])]

    return models


if __name__ == "__main__":
    models = fetch_groq_models()
    print("Available Groq Models:")
    for m in models:
        print(" -", m)

