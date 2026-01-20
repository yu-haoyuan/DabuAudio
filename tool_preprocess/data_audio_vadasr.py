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
        # Add model directory to sys.path to handle imports within the model code
        sys.path.append(str(self.model_path))
        
        # Manually register FunASRNano if possible as requested by user instruction
        try:
            print("Attempting to register FunASRNano model...")
            from funasr.models.fun_asr_nano.model import FunASRNano
        except ImportError as e:
            print(f"Could not import FunASRNano directly: {e}")
            pass

        # Try to import FunASRNano from the model directory
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
        # Ensure wav is on the same device as the VAD model
        wav = wav.to(self.device)
        
        # Calculate speech timestamps
        # Aggressive splitting first, then we merge
        speech_timestamps = self.get_speech_timestamps(
            wav, 
            self.vad_model, 
            sampling_rate=self.sample_rate,
            threshold=0.3, # Lower threshold to catch more speech
            min_speech_duration_ms=100, # Catch short utterances
            min_silence_duration_ms=300 
        )
        
        # Load original audio for slicing/merging
        audio = AudioSegment.from_wav(str(audio_path))
        original_filename = audio_path.stem
        
        # Output structure: data_vadasr/speaker_id/original_filename/
        # JSONL structure: data_vadasr/speaker_id/original_filename.jsonl
        speaker_output_dir = self.output_dir / speaker_id / original_filename
        speaker_output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_segments = []
        
        if not speech_timestamps:
            # If no speech detected, maybe process whole file if length matches target?
            # Or just skip. User requirement implies we must process.
            # If file is short (<10s), maybe just keep it as is?
            if len(audio) < 10000:
                 # Treat whole file as one segment
                 speech_timestamps = [{'start': 0, 'end': int(len(audio) * 16)}] # samples
            else:
                 return []

        # Convert samples to ms
        timestamps_ms = [{'start': t['start'] / self.sample_rate * 1000, 
                          'end': t['end'] / self.sample_rate * 1000} 
                         for t in speech_timestamps]
        
        # Advanced merging logic with buffer/cache
        final_segments = []
        buffer_segment = None # {'start': ms, 'end': ms}
        
        TARGET_MIN_DUR = 5000  # 5s
        TARGET_MAX_DUR = 10000 # 10s
        HARD_MAX_DUR = 15000   # 15s absolute max to force cut
        
        for ts in timestamps_ms:
            current_start = ts['start']
            current_end = ts['end']
            current_dur = current_end - current_start
            
            if buffer_segment is None:
                buffer_segment = {'start': current_start, 'end': current_end}
            else:
                buffer_dur = buffer_segment['end'] - buffer_segment['start']
                gap = current_start - buffer_segment['end']
                combined_dur = buffer_dur + gap + current_dur
                
                # Check if merging makes sense
                if combined_dur <= TARGET_MAX_DUR:
                    # Merge it
                    buffer_segment['end'] = current_end
                elif buffer_dur < TARGET_MIN_DUR:
                    # Buffer is too short, MUST merge even if it exceeds target slightly
                    # unless it exceeds hard max
                    if combined_dur < HARD_MAX_DUR:
                        buffer_segment['end'] = current_end
                    else:
                        # Cannot merge without being too long.
                        # Flush buffer (it's short but we have no choice or it was isolated)
                        final_segments.append(buffer_segment)
                        buffer_segment = {'start': current_start, 'end': current_end}
                else:
                    # Buffer is already good size, flush it
                    final_segments.append(buffer_segment)
                    buffer_segment = {'start': current_start, 'end': current_end}
        
        # Flush remaining buffer
        if buffer_segment:
            final_segments.append(buffer_segment)
            
        # Second pass: check for any remaining tiny segments that could be merged backwards?
        # The forward pass handles most.
        # Check if any segment is < 0.5s and isolated?
        # User said: "if < 0.5s and surrounding 10s is silence, add to cache..."
        # My logic above merges across silence if resulting duration < 10s.
        
        for seg in final_segments:
            start_ms = int(seg['start'])
            end_ms = int(seg['end'])
            duration = end_ms - start_ms
            
            if duration < 100: continue # Skip artifacts
            
            # Export segment
            chunk = audio[start_ms:end_ms]
            out_name = f"{original_filename}_{start_ms}_{end_ms}.wav"
            out_path = speaker_output_dir / out_name
            
            chunk.export(out_path, format="wav")
            
            generated_segments.append({
                'path': out_path,
                'duration_ms': duration,
                'original_file': original_filename,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'speaker_id': speaker_id
            })
            
        return generated_segments

    def run_asr(self, audio_path):
        if self.use_automodel:
            # Optimize generation parameters
            # use_cache=True, beam_size=1 (greedy), max_new_tokens=200
            res = self.asr_model.generate(
                input=[str(audio_path)],
                batch_size=1,
                language="中文",
                itn=True,
                use_cache=True,
                max_new_tokens=256
            )
            # The result format from AutoModel depends on the model type
            # For FunASR-Nano which is likely LLM-based (Qwen), it returns a list of dicts
            # res[0] might contain 'text'
            if isinstance(res, list) and len(res) > 0:
                 if isinstance(res[0], dict) and "text" in res[0]:
                     return res[0]["text"]
                 return str(res[0])
            return str(res)
        else:
            # Using the FunASRNano interface provided by user
            res = self.asr_m.inference(data_in=[str(audio_path)], **self.asr_kwargs)
            return res[0][0]["text"]

    def run(self):
        extensions = {'.wav'}
        files_to_process = []
        
        # Walk through input directory
        # Input structure: data/data_denoise/speaker_id/file.wav
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    files_to_process.append(Path(root) / file)
        
        print(f"Found {len(files_to_process)} input files.")
        
        # Step 1: VAD and Segmentation
        print("Starting VAD segmentation...")
        all_segments_map = {} # Map by (speaker_id, original_file) -> list of segments
        all_segments_flat = []
        
        for file_path in tqdm(files_to_process):
            # Infer speaker_id from parent directory name
            speaker_id = file_path.parent.name
            original_filename = file_path.stem
            
            segments = self.process_vad(file_path, speaker_id)
            
            key = (speaker_id, original_filename)
            if key not in all_segments_map:
                all_segments_map[key] = []
            all_segments_map[key].extend(segments)
            all_segments_flat.extend(segments)
            
        if not all_segments_flat:
            print("No segments generated.")
            return

        # Calculate average duration
        total_duration = sum(seg['duration_ms'] for seg in all_segments_flat)
        avg_duration = total_duration / len(all_segments_flat)
        print(f"\nAverage segment duration: {avg_duration:.2f} ms ({avg_duration/1000:.2f} s)")
        print(f"Total segments: {len(all_segments_flat)}\n")
        
        # Step 2: ASR
        print("Starting ASR transcription...")
        
        # Output ASR results per original file: data_vadasr/speaker_id/original_filename.jsonl
        for (speaker_id, original_filename), segments in all_segments_map.items():
            jsonl_filename = f"{original_filename}.jsonl"
            jsonl_output_path = self.output_dir / speaker_id / jsonl_filename
            
            # Ensure dir exists (already created in VAD step, but good to be safe)
            jsonl_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing {len(segments)} segments for {speaker_id}/{original_filename}...")
            
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
    PROJECT_ROOT = "/data/yhy/code/DabuAudio"
    INPUT_DIR = "data/data_denoise"
    OUTPUT_DIR = "data/data_vadasr"
    MODEL_PATH = "models/Fun-ASR-Nano-2512"
    
    processor = AudioProcessor(PROJECT_ROOT, INPUT_DIR, OUTPUT_DIR, MODEL_PATH)
    processor.run()
