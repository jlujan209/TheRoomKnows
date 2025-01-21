import whisper
from pyannote.audio import Pipeline
import os
import sys
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

def transcribe_and_diarize(audio_file, output_file):
    # Load Pyannote diarization pipeline
    print("Loading Pyannote pipeline...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

    # Load Whisper model
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")

    # Transcribe audio
    print("Transcribing audio with Whisper...")
    transcription = whisper_model.transcribe(audio_file)

    
    # Perform speaker diarization
    print("Performing speaker diarization...")
    diarization = diarization_pipeline(audio_file)

    # Match transcription with diarization results
    results = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end

        # Find corresponding transcription segment
        matching_segments = [
            entry["text"] for entry in transcription["segments"]
            if entry["start"] >= start_time and entry["end"] <= end_time
        ]

        if matching_segments:
            text = " ".join(matching_segments)
            results.append((speaker, text))

    # Write results to the output file
    with open(output_file, "w") as f:
        for speaker, text in results:
            f.write(f"{speaker}: {text}\n")

    print(f"Transcription with speakers written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <audio_file> <output_file>")
        sys.exit(1)

    print(f"hf token: {hf_token}")

    audio_file = sys.argv[1]
    output_file = sys.argv[2]

    '''
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found.")
        sys.exit(1)
'''
    transcribe_and_diarize(audio_file, output_file)
