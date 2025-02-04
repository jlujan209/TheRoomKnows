import os
import json
from google.cloud import speech

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Lujan/Documents/GitHub/TheRoomKnows/GoogleSpeechAnalysis/serviceaccountkey.json"

# Path to an existing audio file (must be in LINEAR16 format)
AUDIO_FILE_PATH = "C:/Users/Lujan/Documents/GitHub/TheRoomKnows/sampleaudio.wav"

# Audio file properties
RATE = 16000  # Ensure the file is sampled at 16 kHz

def transcribe_audio(file_path):
    """Transcribes an existing audio file and performs speaker diarization."""

    client = speech.SpeechClient()

    # Read the audio file
    with open(file_path, "rb") as audio_file:
        audio_content = audio_file.read()

    audio = speech.RecognitionAudio(content=audio_content)

    # Configure speech recognition with diarization
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        model="video",
        use_enhanced=True,
        diarization_config=speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=1,
            max_speaker_count=5,
        ),
    )

    print("Transcribing audio...")

    response = client.recognize(config=config, audio=audio)

    # Save raw API response for debugging
    with open("raw_output.json", "w") as raw_file:
        raw_data = response.__class__.to_dict(response)
        json.dump(raw_data, raw_file, indent=4)

    # Process speaker diarization
    speaker_transcripts = {}
    
    for result in response.results:
        words = result.alternatives[0].words

        for word_info in words:
            speaker = word_info.speaker_tag
            word = word_info.word

            if speaker not in speaker_transcripts:
                speaker_transcripts[speaker] = []
            
            speaker_transcripts[speaker].append(word)

    # Save formatted transcription
    with open("transcription_output.txt", "w") as text_file:
        for speaker, words in sorted(speaker_transcripts.items()):
            if speaker == 0:
                continue
            transcript = " ".join(words)
            print(f"Speaker {speaker}: {transcript}")
            text_file.write(f"Speaker {speaker}: {transcript}\n")

    print("Transcription saved to transcription_output.txt")
    print("Raw API response saved to raw_output.json")

if __name__ == "__main__":
    transcribe_audio(AUDIO_FILE_PATH)
