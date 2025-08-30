# Speaker_Diarization_Using_Pyhton
Whisper Diarizer is a speech-to-text and speaker diarization tool powered by OpenAI’s Whisper and Pyannote. It transcribes audio, identifies speakers, and generates structured text/JSON outputs. Useful for interviews, meetings, podcasts, and multi-speaker conversations.

WhisperDiarizer 🎙️

WhisperDiarizer is a Python-based speech-to-text and speaker diarization tool.
It combines OpenAI Whisper for transcription with Pyannote embeddings and clustering for speaker identification.
This helps separate speakers and generate accurate transcripts from interviews, podcasts, meetings, or any multi-speaker audio.

Features
(1)Convert audio to mono WAV automatically
(2)Transcribe audio using Whisper (tiny to large models)
(3)Speaker diarization with clustering (auto or fixed number of speakers)
(4)Supports JSON and text output formats
(5)Handles multiple audio formats with Pydub and Torchaudio

Runs on CPU or GPU

Installation
git clone https://github.com/your-username/WhisperDiarizer.git
cd WhisperDiarizer
pip install -r requirements.txt

Usage
python diarizer.py path/to/audio.mp3 -m small -f json -o output.json


Options:

-n / --num-speakers → Fixed number of speakers (auto-detected if not set)
-m / --model → Whisper model size (tiny, base, small, medium, large)
-f / --format → Output format (json or text)
-o / --output → Save results to a file

Example Output
JSON:

[
  {
    "speaker": "SPEAKER 1",
    "start": 0.0,
    "end": 5.2,
    "text": "Hello, welcome to the meeting."
  },
  {
    "speaker": "SPEAKER 2",
    "start": 5.3,
    "end": 8.1,
    "text": "Thanks, happy to be here."
  }
]


Text:

SPEAKER 1 00:00:00
Hello, welcome to the meeting.

SPEAKER 2 00:00:05
Thanks, happy to be here.


Requirements
->Python 3.8+
->OpenAI Whisper
->Pyannote Audio
->Pydub
->Torchaudio
->Scikit-learn
->NumPy


Install dependencies:
-> pip install -r requirements.txt

Use Cases
(1) Meeting transcription and speaker separation
(2) Podcast or interview processing
(3) Multi-speaker research datasets
(4)Lecture or event transcription

License
MIT License – free to use and modify.

Contributing
-> Pull requests are welcome.
-> If you’d like to add features (e.g., GUI, diarization improvements, streaming support), feel free to fork and contribute.

Pull requests are welcome.
If you’d like to add features (e.g., GUI, diarization improvements, streaming support), feel free to fork and contribute.
