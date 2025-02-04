import os
import wave
import pyaudio

# Audio recording parameters
RATE = 16000  # Sample rate (16 kHz for LINEAR16 format)
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Format for Linear16 (16-bit signed integers)
CHANNELS = 1  # Mono audio
OUTPUT_WAV_FILE = "sampleaudio.wav"  # Output file name

def record_audio():
    """Records audio from the microphone and saves it as a .wav file."""
    audio_interface = pyaudio.PyAudio()

    # Open the audio stream
    stream = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("Recording... (Press Ctrl+C to stop)")

    frames = []

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred while recording: {e}")
    finally:
        print("Recording stopped.")
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()

    # Save audio to a .wav file in Linear16 format
    with wave.open(OUTPUT_WAV_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    print(f"Audio saved to {OUTPUT_WAV_FILE}")

if __name__ == "__main__":
    record_audio()
