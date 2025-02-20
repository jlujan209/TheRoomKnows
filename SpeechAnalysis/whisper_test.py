import whisper
import json

def transcribe(in_file, out_file):
    model = whisper.load_model('base')
    result = model.transcribe(in_file)
    with open(out_file, 'w') as f:
        json.dump(result, f)