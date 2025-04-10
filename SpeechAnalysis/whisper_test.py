import whisper
import time
import json

model = whisper.load_model('base')

s = time.time()
result = model.transcribe('videoplayback.m4a')
e = time.time()
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
print(result)
print(f"Time taken: {e - s} seconds")
