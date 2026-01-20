import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def normalize_audio(src_root, dst_root, ffmpeg_path):
    src_path = Path(src_root)
    dst_path = Path(dst_root)
    
    # Extensions to look for
    extensions = {'.wav', '.aac', '.mp3', '.m4a', '.flac'}
    
    files_to_process = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                files_to_process.append(Path(root) / file)
    
    print(f"Found {len(files_to_process)} audio files in {src_root}.")
    
    if not files_to_process:
        print("No audio files found. Exiting.")
        return

    success_count = 0
    fail_count = 0

    for file_path in tqdm(files_to_process):
        try:
            # Construct output path with flattening logic
            rel_path = file_path.relative_to(src_path)
            parts = rel_path.parts
            
            # If we have at least one subdirectory (e.g. hxx/...), keep the first one as speaker ID
            # and append the filename, ignoring intermediate directories.
            if len(parts) >= 2:
                speaker_dir = parts[0]
                filename = parts[-1]
                out_rel_path = Path(speaker_dir) / filename
            else:
                # File is directly in src_root
                out_rel_path = Path(parts[-1])
            
            out_file_path = dst_path / out_rel_path
            out_file_path = out_file_path.with_suffix('.wav')
            
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # FFmpeg command: -i input -ar 16000 -ac 1 output.wav
            # Use -y to overwrite
            # Use -loglevel error to reduce noise
            cmd = [
                ffmpeg_path,
                '-y',
                '-i', str(file_path),
                '-ar', '16000',
                '-ac', '1',
                '-loglevel', 'error',
                str(out_file_path)
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error converting {file_path}: {e.stderr.decode()}")
            fail_count += 1
        except Exception as e:
            print(f"Unexpected error processing {file_path}: {e}")
            fail_count += 1

    print(f"Processing complete.")
    print(f"Successfully normalized: {success_count}")
    print(f"Failed: {fail_count}")

import sys
import shutil

if __name__ == "__main__":
    # Configuration
    src_dir = "/data/yhy/code/DabuAudio/data/data_vanilla"
    dst_dir = "/data/yhy/code/DabuAudio/data/data_norm"
    
    # Try to find ffmpeg in the current conda environment first
    ffmpeg_bin = os.path.join(sys.prefix, 'bin', 'ffmpeg')
    
    if not os.path.exists(ffmpeg_bin):
        # Fallback to system path
        ffmpeg_bin = shutil.which('ffmpeg')
        
    if not ffmpeg_bin:
        print("Error: FFmpeg not found. Please install it with 'conda install ffmpeg' or 'apt install ffmpeg'.")
        exit(1)

    print(f"Using FFmpeg: {ffmpeg_bin}")
    normalize_audio(src_dir, dst_dir, ffmpeg_bin)
