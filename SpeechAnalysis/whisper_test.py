import whisper
import time

model = whisper.load_model('base')

s = time.time()
result = model.transcribe('videoplayback.m4a')
e = time.time()
print(result)
print(f"Time taken: {e - s} seconds")
