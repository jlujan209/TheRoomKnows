import sounddevice as sd
import soundfile as sf
import threading

def record_audio(filename, stop_event, device, channels, samplerate=48000):
    print("Recording audio...")
    with sf.SoundFile(filename, mode='w', samplerate=samplerate, channels=channels) as file:
        def callback(indata, frames, time, status):
            if stop_event.is_set():
                raise sd.CallbackStop()
            file.write(indata)

        with sd.InputStream(samplerate=samplerate, device=device, channels=channels, callback=callback):
            stop_event.wait()  # Wait until the stop event is triggered
    print(f"Audio saved to {filename}")

# List available audio input devices
def list_audio_devices():
    print("Available host apis:")
    apis = sd.query_hostapis()
    for idx, api in enumerate(apis):
        print(f"{idx}: {api['name']}")
    print("\nAvailable audio input devices:\n")
    devices = sd.query_devices()
    input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]
    for idx, dev in enumerate(input_devices):
        print(f"{idx}: {dev['name']} - {dev['hostapi']}")
    return input_devices

stop_event = threading.Event()

devices = list_audio_devices()
# x = input("Select input device and press Enter\n\t1) NVIDIA Broadcast\n\t2) USB PnP Audio Device\n")
# print(f"You selected {x}")
device_name = "Microphone (NVIDIA Broadcast)"
device_name = "Line (AudioBox USB 96)"

# Find the device index by name
# devices = sd.query_devices()
hostapi = sd.query_hostapis()
# get index of WASAPI hostapi
wasapi_index = next((idx for idx, api in enumerate(hostapi) if api['name'] == 'Windows WASAPI'), None)
if wasapi_index is None:
    print("WASAPI host API not found.")
    exit(1)

device = next((idx for idx, dev in enumerate(devices) if (dev['name'] == device_name and dev['hostapi'] == wasapi_index)), None)
print(device)
if device is None:
    print(f"Device '{device_name}' not found.")
else:
    print("Press 'q' to stop recording.")
    threading.Thread(target=record_audio, args=("NVIDIA_test.wav", stop_event, device, 1), daemon=True).start()

    while True:
        if input().strip().lower() == 'q':
            stop_event.set()
            break

    print(f"Audio saved to NVIDIA_test.wav")