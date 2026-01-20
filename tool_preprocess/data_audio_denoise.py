import os
import sys
from pathlib import Path
from tqdm import tqdm
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def denoise_audio(project_root, src_rel_path, dst_rel_path, model_rel_path):
    project_path = Path(project_root).resolve()
    src_path = project_path / src_rel_path
    dst_path = project_path / dst_rel_path
    model_path = project_path / model_rel_path

    print(f"Project Root: {project_path}")
    print(f"Source Directory: {src_path}")
    print(f"Destination Directory: {dst_path}")
    print(f"Model Path: {model_path}")

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Initialize the denoising pipeline with the local model
    try:
        ans = pipeline(
            Tasks.acoustic_noise_suppression,
            model=str(model_path)
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)

    # Extensions to look for
    extensions = {'.wav'}
    
    files_to_process = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                files_to_process.append(Path(root) / file)
    
    print(f"Found {len(files_to_process)} audio files in {src_path}.")
    
    if not files_to_process:
        print("No audio files found. Exiting.")
        return

    success_count = 0
    fail_count = 0

    for file_path in tqdm(files_to_process):
        try:
            # Construct output path preserving structure relative to src_path
            rel_path = file_path.relative_to(src_path)
            out_file_path = dst_path / rel_path
            
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run denoising
            # The pipeline expects input path and output path
            ans(str(file_path), output_path=str(out_file_path))
            
            success_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            fail_count += 1

    print(f"Processing complete.")
    print(f"Successfully denoised: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    # Define paths relative to the script execution or project root
    # Assuming script is run from project root or paths are relative to project root
    
    # We will assume the project root is /data/yhy/code/DabuAudio as requested
    PROJECT_ROOT = "/data/yhy/code/DabuAudio"
    
    # Relative paths
    SRC_REL_DIR = "data/data_norm"
    DST_REL_DIR = "data/data_denoise"
    MODEL_REL_PATH = "models/speech_zipenhancer_ans_multiloss_16k_base"
    
    denoise_audio(PROJECT_ROOT, SRC_REL_DIR, DST_REL_DIR, MODEL_REL_PATH)
