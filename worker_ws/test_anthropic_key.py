import requests
import json
import os

url = "https://api.anthropic.com/v1/messages"

# Ensure you are using your valid API key here
api_key = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")

headers = {
    "Content-Type": "application/json",
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01"
}

data = {
    # We are testing with Haiku, which is the most compatible model
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Test connection."}]
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("Success! Your API key works with Haiku.")
    print(response.json()['content'][0]['text'])
else:
    print(f"Error {response.status_code}: {response.text}")