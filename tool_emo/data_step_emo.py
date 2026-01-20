import os
import sys
import json
from tqdm import tqdm

# Add Step-Audio2 to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'Step-Audio2'))

from stepaudio2 import StepAudio2

def analyze_emotion(model, audio_path):
    if not os.path.exists(audio_path):
        print(f"Audio not found: {audio_path}")
        return None

    messages = [
        {"role": "system", "content": "你是一位情感分析专家。请分析这段语音中说话人的情感状态，需要详细解释一下为什么，你是从什么地方捕捉到这个情感的"},
        {"role": "human", "content": [{"type": "audio", "audio": audio_path}]},
        {"role": "assistant", "content": None}
    ]
    
    try:
        _, text, _ = model(messages, max_new_tokens=256, temperature=0.5, top_p=0.9, do_sample=True)
        return text
    except Exception as e:
        print(f"Model inference error: {e}")
        return None

def process_files(input_root, output_root, model_path):
    # Initialize model
    print(f"Loading model from {model_path}...")
    try:
        model = StepAudio2(model_path)
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Walk through input directory
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_path = os.path.join(root, file)
                
                # Determine output path preserving structure
                rel_path = os.path.relpath(jsonl_path, input_root)
                output_path = os.path.join(output_root, rel_path)
                
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                print(f"Processing {jsonl_path} -> {output_path}")
                
                # Directory where audio files for this jsonl are located
                # Based on structure: .../hxx/01月19日_1.jsonl -> audio in .../hxx/01月19日_1/
                jsonl_basename = os.path.splitext(file)[0]
                audio_dir = os.path.join(root, jsonl_basename)
                
                if not os.path.exists(audio_dir):
                    print(f"Warning: Audio directory {audio_dir} does not exist for {jsonl_path}")
                    # Try current directory as fallback if audio files are not in subdir
                    # audio_dir = root
                
                with open(jsonl_path, 'r', encoding='utf-8') as fin, \
                     open(output_path, 'w', encoding='utf-8') as fout:
                    
                    lines = fin.readlines()
                    for line in tqdm(lines, desc=file):
                        try:
                            line = line.strip()
                            if not line:
                                continue
                            data = json.loads(line)
                            filename = data.get('filename')
                            
                            emo_result = None
                            if filename:
                                # Construct audio path
                                # Try constructed path first
                                audio_path = os.path.join(audio_dir, filename)
                                if not os.path.exists(audio_path):
                                     # Try checking if filename already contains path or is relative to root
                                     # But based on observation, it is filename only.
                                     pass

                                # Analyze emotion
                                emo_result = analyze_emotion(model, audio_path)
                                
                            if emo_result:
                                data['emo'] = emo_result
                            else:
                                # Keep it empty or mark as failed? User said "value为字符串".
                                # If failed, maybe don't add key or add empty string?
                                # I'll add a placeholder if analysis fails to indicate it was processed.
                                # But typically if it fails we might want to retry or skip.
                                # I'll just skip adding the key if it fails, or maybe add "Unknown".
                                # User said "add a key: emo".
                                data['emo'] = emo_result if emo_result else ""
                            
                            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                            fout.flush()
                        except Exception as e:
                            print(f"Error processing line: {e}")
                            # Write original data if possible, but we might have modified it.
                            pass

if __name__ == '__main__':
    input_dir = '/data/xiaobu/DabuAudio/data/data_vadasr'
    output_dir = '/data/xiaobu/DabuAudio/data/data_emo'
    model_dir = '/data/xiaobu/DabuAudio/models/Step-Audio-2-mini'
    
    process_files(input_dir, output_dir, model_dir)
