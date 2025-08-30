import whisper #Open AI Whisper library for speech to text 
import torch 
import wave
import contextlib
import json
import datetime
import traceback
import numpy as np  
from pathlib import Path
from pydub import AudioSegment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from typing import Union, Optional, List, Dict
import torchaudio

class AudioDiarizer:
    def __init__(self, whisper_model: str = "small", device: Optional[str] = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Loading Whisper model '{whisper_model}' on {self.device}...")
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        
        print("[INFO] Loading speaker embedding model...")
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device(self.device)
        )
        self.audio = Audio()
    
    def convert_to_wav(self, audio_path: Union[str, Path]) -> Path:
        #Convert any audio file to mono WAV format
        audio_path = Path(audio_path)
        if audio_path.suffix.lower() == '.wav':
            # Verify if existing WAV is mono
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] == 1:  # Already mono
                return audio_path
            
        print(f"[INFO] Converting {audio_path.suffix} to mono WAV...")
        audio = AudioSegment.from_file(audio_path)
        
        # Force mono conversion if needed
        if audio.channels > 1:
            audio = audio.set_channels(1)
            
        wav_path = audio_path.with_suffix(".wav")
        audio.export(wav_path, format="wav", parameters=["-ac", "1"])  # Ensure mono
        return wav_path
    
    def get_duration(self, path: Union[str, Path]) -> float:
        """Get audio duration in seconds"""
        with contextlib.closing(wave.open(str(path), 'r')) as f:
            return f.getnframes() / float(f.getframerate())
    
    def segment_embedding(self, path: Union[str, Path], segment: Dict, duration: float) -> np.ndarray:
        """Create embedding for a single segment with mono audio check"""
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        
        # Load and verify mono
        waveform, sample_rate = self.audio.crop(str(path), clip)
        if waveform.shape[0] > 1:  # If stereo
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        
        return self.embedding_model(waveform[None])
    
    def make_embeddings(self, path: Union[str, Path], segments: List[Dict], duration: float) -> np.ndarray:
        """Create speaker embeddings for all segments with error handling"""
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            try:
                embeddings[i] = self.segment_embedding(path, segment, duration)
            except Exception as e:
                print(f"[WARNING] Failed to process segment {i}: {str(e)}")
                embeddings[i] = np.zeros(192)  # Fallback zero vector
        return np.nan_to_num(embeddings)
    
    def estimate_num_speakers(self, embeddings: np.ndarray, max_speakers: int = 10) -> int:
        """Automatically estimate the optimal number of speakers"""
        n_samples = len(embeddings)
        if n_samples < 3:
            return 1
            
        best_score = -1
        best_k = 1
        
        for k in range(2, min(max_speakers, n_samples - 1) + 1):
            clustering = AgglomerativeClustering(n_clusters=k).fit(embeddings)
            labels = clustering.labels_
            if len(set(labels)) < 2:
                continue
            try:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except ValueError:
                continue
        
        print(f"[INFO] Estimated {best_k} speakers (silhouette score: {best_score:.2f})")
        return best_k
    
    def add_speaker_labels(self, segments: List[Dict], embeddings: np.ndarray, num_speakers: Optional[int] = None):
        """Add speaker labels to segments using clustering"""
        if num_speakers is None:
            num_speakers = self.estimate_num_speakers(embeddings)
        
        if len(segments) == 1:
            segments[0]['speaker'] = 'SPEAKER 1'
            return
            
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        for i, label in enumerate(clustering.labels_):
            segments[i]["speaker"] = f'SPEAKER {label + 1}'
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        return str(datetime.timedelta(seconds=round(seconds)))
    
    def get_text_output(self, segments: List[Dict]) -> str:
        """Format output as human-readable text"""
        output = []
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                if i != 0:
                    output.append('')  # Add empty line between speakers
                output.append(f"{segment['speaker']} {self.format_timestamp(segment['start'])}")
                output.append('')
            output.append(segment['text'].strip())
        return '\n'.join(output)
    
    def get_json_output(self, segments: List[Dict]) -> List[Dict]:
        """Format output as JSON-compatible list of dicts"""
        output = []
        last_speaker = None
        last_text = None
        
        for segment in segments:
            speaker = segment.get("speaker", "SPEAKER ?")
            text = segment["text"].strip()
            if text == last_text and speaker == last_speaker:
                continue
            output.append({
                "speaker": speaker,
                "start": segment["start"],
                "end": segment["end"],
                "text": text
            })
            last_speaker = speaker
            last_text = text
        
        return output
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        output_format: str = "json"
    ) -> Union[str, List[Dict]]:
       
        try:
            # Convert to mono WAV if needed
            wav_path = self.convert_to_wav(audio_path)
            print(f"[INFO] Processing file: {wav_path}")
            
            # Check duration
            duration = self.get_duration(wav_path)
            print(f"[INFO] Duration: {duration:.2f} seconds")
            if duration > 4 * 60 * 60:  # 4 hours
                raise ValueError("Audio duration too long (max 4 hours)")
            
            # Transcribe with Whisper
            print("[INFO] Starting transcription...")
            result = self.whisper_model.transcribe(str(wav_path))
            segments = result["segments"]
            print(f"[INFO] Transcription complete ({len(segments)} segments)")
            
            # Speaker diarization
            if len(segments) > 1:
                embeddings = self.make_embeddings(wav_path, segments, duration)
                self.add_speaker_labels(segments, embeddings, num_speakers)
            
            # Return requested output format
            if output_format.lower() == "text":
                return self.get_text_output(segments)
            return self.get_json_output(segments)
            
        except Exception as e:
            print("[ERROR] Processing failed:")
            traceback.print_exc()
            raise RuntimeError(f"Error processing audio: {str(e)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Diarization with Whisper")
    parser.add_argument("audio_path", help="Path to the audio file to process")
    parser.add_argument("-n", "--num-speakers", type=int, default=None,
                        help="Fixed number of speakers (auto-detected if not specified)")
    parser.add_argument("-m", "--model", default="small",
                        help="Whisper model size (tiny, base, small, medium, large, large-v2)")
    parser.add_argument("-f", "--format", choices=["json", "text"], default="json",
                        help="Output format")
    parser.add_argument("-o", "--output", help="Output file path")
    
    args = parser.parse_args()
    
    try:
        diarizer = AudioDiarizer(whisper_model=args.model)
        result = diarizer.process_audio(
            args.audio_path,
            num_speakers=args.num_speakers,
            output_format=args.format
        )
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.format == "json":
                    json.dump(result, f, indent=4, ensure_ascii=False)
                else:
                    f.write(result)
            print(f"[INFO] Results saved to {args.output}")
        else:
            print("\nResults:")
            print("=" * 50)
            if args.format == "json":
                print(json.dumps(result, indent=4, ensure_ascii=False))
            else:
                print(result)
            print("=" * 50)
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()