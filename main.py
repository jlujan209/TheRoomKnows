import threading
import FacialAnalysis.facial_analysis
import SpeechAnalysis
import FacialAnalysis
import SpeechAnalysis.capture_audio
import SpeechAnalysis.whisper_test

stop_event = threading.Event()

audio_devices = SpeechAnalysis.capture_audio.list_audio_devices()
print("Select the microphone device to use: ", end='')
for idx, dev in enumerate(audio_devices):
    print(f"{idx}: {dev['name']} (Max Channels: {dev['max_input_channels']})")
dev_index = input()
microphone = audio_devices[int(dev_index)]
channels = microphone["max_input_channels"]

image_thread = threading.Thread(target=FacialAnalysis.facial_analysis.main, args=(stop_event,))
audio_thread = threading.Thread(target=SpeechAnalysis.capture_audio.record_audio, args=("SpeechAnalysis/output.wav", stop_event, microphone["name"], channels, 44100))

image_thread.start()
audio_thread.start()

input("Press enter to stop recording and image collection")
stop_event.set()

image_thread.join()
audio_thread.join()

SpeechAnalysis.whisper_test.transcribe("SpeechAnalysis/output.wav", "SpeechAnalysis/output.json")

print("DONE")