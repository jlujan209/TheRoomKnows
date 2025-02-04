# Text-to-Speech and Speaker Diarization using Google Cloud API
## Setup
In order to run this a service account json key is needed for GCP that has the Speech-to-text API enabled.

Steps to create the json key:
1. Go to IAM & Security in GCP.
2. Go to Service Accounts.
3. Create new Service Account and set the role to speech administrator.
4. Click on the newly created service account.
5. Go to Keys.
6. Create a new json key, it should be downloaded automatically.
7. Place this new key in this directory and update the path to this key in the SpeechAnalysis.py file.
8. Make sure that the Speech-to-text API is enabled.

## Running the Speech Analysis

To run the speech analysis:
1. Run AudioRecorder.py, this will start recording audio into the proper format (LINEAR16 .wav) to be used for the transcription, (To stop the recording use CTRL ^C)
2. Run SpeechAnalysis.py (Before running make sure that the path to the .wav file created by AudioRecorder.py is correct in SpeechAnalysis.py)