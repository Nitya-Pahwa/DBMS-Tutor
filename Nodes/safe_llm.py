"""
Module: safe_llm.py

Purpose:
Wrapper around Groq LLM API to safely generate responses.

Why:
Handles API errors gracefully and standardizes LLM calls.

Function:
safe_invoke(prompt) -> returns model response
"""

import os
from openai import OpenAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client using OpenAI-compatible API
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Wrapper function to safely call LLM API
def safe_invoke(prompt: str) -> str:
    try:
        # Send prompt to LLaMA model hosted on Groq
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast + free on Groq
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(" GROQ API ERROR:", repr(e))
        return "Answer could not be generated."
