'''
Attempts to split text into questions and answers rather than breaking it up by speaker.
'''
import openai
import json
from dotenv import load_dotenv
import os

load_dotenv()
KEY = os.getenv

client = OpenAI(api_key=KEY)

model = "gpt-4o-mini"
with open("test.json", "r") as f:
    text = json.load(f)["text"]
# Define the prompt
prompt = (
    "Your task is to take a conversation between a doctor and a patient and split it "
    "into question-answer pairs. Each question-answer pair should be separated by a '|'."
)

# Interact with the API
diarized_text = client.chat(
    model=model,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
)

print(diarized_text)

with open("output.json", "w") as f:
    json.dump(diarized_text, f)