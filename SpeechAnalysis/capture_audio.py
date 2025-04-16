import sounddevice as sd
# import soundfile as sf

def record_audio(filename, stop_event, device, channels, samplerate=44100):
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
    print("\nAvailable audio input devices:\n")
    devices = sd.query_devices()
    input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]
    for idx, dev in enumerate(input_devices):
        print(f"{idx}: {dev['name']} (Max Channels: {dev['max_input_channels']})")
    return input_devices

if __name__ == "__main__":
    list_audio_devices()