import os
import sys
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
import importlib.util
from funasr import AutoModel

def load_local_module(module_path, module_name):
    """Load a python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

class AudioProcessor:
    def __init__(self, project_root, input_dir, output_dir, model_path):
        self.project_root = Path(project_root).resolve()
        self.input_dir = self.project_root / input_dir
        self.output_dir = self.project_root / output_dir
        self.model_path = self.project_root / model_path
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        
        # Add ffmpeg to PATH for pydub
        ffmpeg_bin_dir = os.path.join(sys.prefix, 'bin')
        if os.path.exists(os.path.join(ffmpeg_bin_dir, 'ffmpeg')):
             os.environ["PATH"] += os.pathsep + ffmpeg_bin_dir

        # Load VAD model
        print("Loading Silero VAD model...")
        try:
            from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
            self.vad_model = load_silero_vad()
            self.get_speech_timestamps = get_speech_timestamps
            self.read_audio = read_audio
        except ImportError:
            print("Failed to import silero_vad. Please install it with 'pip install silero-vad'.")
            sys.exit(1)
            
        self.vad_model.to(self.device)
        
        # Load ASR model
        print(f"Loading FunASR model from {self.model_path}...")
        sys.path.append(str(self.model_path))
        
        try:
            print("Attempting to register FunASRNano model...")
            from funasr.models.fun_asr_nano.model import FunASRNano
        except ImportError as e:
            print(f"Could not import FunASRNano directly: {e}")
            pass

        print("Using funasr.AutoModel for Fun-ASR-Nano-2512...")
        try:
            self.asr_model = AutoModel(
                model=str(self.model_path),
                trust_remote_code=True,
                device=self.device,
                disable_update=True
            )
            self.use_automodel = True
        except Exception as e:
            print(f"Failed to load model with AutoModel: {e}")
            sys.exit(1)

    def process_vad(self, audio_path, speaker_id):
        wav = self.read_audio(str(audio_path), sampling_rate=self.sample_rate)
        wav = wav.to(self.device)
        
        # 1. Extract VAD segments
        # Increased sensitivity to split fine segments: 
        # min_silence_duration_ms=50 (aggressive split on silence)
        # min_speech_duration_ms=50 (keep short words)
        speech_timestamps = self.get_speech_timestamps(
            wav, 
            self.vad_model, 
            sampling_rate=self.sample_rate,
            threshold=0.4, # Slightly more sensitive to speech
            min_speech_duration_ms=50, 
            min_silence_duration_ms=50 
        )
        
        # Load original audio for slicing
        audio = AudioSegment.from_wav(str(audio_path))
        original_filename = audio_path.stem
        
        # Output dir: data_vadasr/hxx_v2/original_filename/
        target_speaker_dir_name = f"{speaker_id}_v2"
        speaker_output_dir = self.output_dir / target_speaker_dir_name / original_filename
        speaker_output_dir.mkdir(parents=True, exist_ok=True)
        
        if not speech_timestamps:
            return []

        # Convert samples to ms
        timestamps_ms = [{'start': t['start'] / self.sample_rate * 1000, 
                          'end': t['end'] / self.sample_rate * 1000} 
                         for t in speech_timestamps]
        
        generated_segments = []
        
        # 2. Buffer splicing strategy
        # Accumulate AUDIO CONTENT (skipping silence) until > 5s
        
        buffer_audio = AudioSegment.empty()
        buffer_start_ms = -1
        buffer_end_ms = -1
        
        # Helper to flush buffer
        def flush_buffer(buf_audio, start, end, segments_list):
            if len(buf_audio) < 100: # Skip if empty or too short
                return
            
            out_name = f"{original_filename}_{int(start)}_{int(end)}.wav"
            out_path = speaker_output_dir / out_name
            
            buf_audio.export(out_path, format="wav")
            
            segments_list.append({
                'path': out_path,
                'duration_ms': len(buf_audio),
                'original_file': original_filename,
                'start_ms': start, # Note: These are range markers, not continuous time in file
                'end_ms': end,
                'speaker_id': target_speaker_dir_name,
                'jsonl_dir': speaker_output_dir # Store dir to save jsonl later
            })

        for seg in timestamps_ms:
            seg_start = seg['start']
            seg_end = seg['end']
            
            # Cut the speech segment
            chunk = audio[seg_start:seg_end]
            
            if buffer_start_ms == -1:
                buffer_start_ms = seg_start
            
            buffer_audio += chunk
            buffer_end_ms = seg_end # Update end marker
            
            if len(buffer_audio) >= 5000: # 5 seconds
                flush_buffer(buffer_audio, buffer_start_ms, buffer_end_ms, generated_segments)
                # Reset buffer
                buffer_audio = AudioSegment.empty()
                buffer_start_ms = -1
                buffer_end_ms = -1
                
        # Flush remaining
        if len(buffer_audio) > 0:
             flush_buffer(buffer_audio, buffer_start_ms, buffer_end_ms, generated_segments)
            
        return generated_segments

    def run_asr(self, audio_path):
        if self.use_automodel:
            torch.cuda.empty_cache()
            try:
                res = self.asr_model.generate(
                    input=[str(audio_path)],
                    batch_size=1,
                    language="中文",
                    itn=True,
                    use_cache=True,
                    max_new_tokens=128
                )
            except torch.OutOfMemoryError:
                 print(f"OOM encountered. Skipping ASR.")
                 torch.cuda.empty_cache()
                 return ""
            
            if isinstance(res, list) and len(res) > 0:
                 if isinstance(res[0], dict) and "text" in res[0]:
                     return res[0]["text"]
                 return str(res[0])
            return str(res)
        return ""

    def run(self):
        extensions = {'.wav'}
        files_to_process = []
        
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    files_to_process.append(Path(root) / file)
        
        print(f"Found {len(files_to_process)} input files.")
        
        print("Starting VAD segmentation and splicing...")
        all_segments_map = {} # Map by (speaker_id_v2, original_filename, jsonl_dir) -> list of segments
        all_segments_flat = []
        
        for file_path in tqdm(files_to_process):
            speaker_id = file_path.parent.name
            original_filename = file_path.stem
            
            segments = self.process_vad(file_path, speaker_id)
            
            if not segments:
                continue
                
            # Use the info from the first segment to identify the group
            # Assuming all segments from one file go to the same output dir
            jsonl_dir = segments[0]['jsonl_dir']
            target_speaker_id = segments[0]['speaker_id']
            
            key = (target_speaker_id, original_filename, jsonl_dir)
            if key not in all_segments_map:
                all_segments_map[key] = []
            all_segments_map[key].extend(segments)
            all_segments_flat.extend(segments)
            
        if not all_segments_flat:
            print("No segments generated.")
            return

        total_duration = sum(seg['duration_ms'] for seg in all_segments_flat)
        avg_duration = total_duration / len(all_segments_flat) if all_segments_flat else 0
        print(f"\nAverage segment duration: {avg_duration:.2f} ms")
        print(f"Total segments: {len(all_segments_flat)}\n")
        
        print("Starting ASR transcription...")
        
        # Output ASR results.
        # process_vad creates: data_vadasr/hxx_v2/original_filename/xxx.wav
        # User requested jsonl in: data_vadasr/hxx_v2/original_filename.jsonl (NOT inside original_filename dir)
        
        for (target_speaker_id, original_filename, jsonl_dir), segments in all_segments_map.items():
            jsonl_filename = f"{original_filename}.jsonl"
            # jsonl_dir points to .../hxx_v2/original_filename/
            # We want .../hxx_v2/
            target_jsonl_dir = jsonl_dir.parent
            jsonl_output_path = target_jsonl_dir / jsonl_filename
            
            print(f"Processing {len(segments)} segments for {target_speaker_id}/{original_filename} -> {jsonl_output_path}")
            
            with open(jsonl_output_path, 'w', encoding='utf-8') as f:
                for seg in tqdm(segments, leave=False):
                    text = self.run_asr(seg['path'])
                    
                    entry = {
                        "filename": seg['path'].name,
                        "duration_sec": round(seg['duration_ms'] / 1000, 3),
                        "text": text,
                        "original_file": seg['original_file'],
                        "start_ms": seg['start_ms'],
                        "end_ms": seg['end_ms']
                    }
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    
        print(f"ASR completed. Results saved in {self.output_dir}")

if __name__ == "__main__":
    PROJECT_ROOT = "/data/xiaobu/DabuAudio"
    INPUT_DIR = "data/data_denoise"
    OUTPUT_DIR = "data/data_vadasr"
    MODEL_PATH = "models/Fun-ASR-Nano-2512"
    
    processor = AudioProcessor(PROJECT_ROOT, INPUT_DIR, OUTPUT_DIR, MODEL_PATH)
    processor.run()
