from google.cloud import speech
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Lujan/Documents/GitHub/TheRoomKnows/GoogleSpeechAnalysis/serviceaccountkey.json"

client = speech.SpeechClient()
print("Authentication successful! Google Speech-to-Text API is ready.")
