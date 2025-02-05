import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_KEY")

# Path to your audio file (supported formats: mp3, wav, m4a, etc.)
audio_file_path = "C:/Users/Lujan/Documents/GitHub/TheRoomKnows/sampleaudio.wav"

# OpenAI now requires explicit API key setting
client = openai.OpenAI(api_key=api_key)

# Open the file in binary mode and transcribe
with open(audio_file_path, "rb") as audio_file:
    response = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        language="en",  
        response_format="text"  
    )

# Print or save the transcription
print(response)
