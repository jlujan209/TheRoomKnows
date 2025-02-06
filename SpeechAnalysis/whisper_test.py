import whisper

model = whisper.load_model('base')
result = model.transcribe('whisper_recording.m4a')

print(result)