import shutil
from pathlib import Path

def process_audio_files(source_base_dir, output_base_dir):
    """
    读取 source_base_dir 下的第一级目录，
    递归查找其子目录中的音频文件并移动到新的结构中，同时避开输出目录。
    """
    source_path = Path(source_base_dir).resolve()
    output_path = Path(output_base_dir).resolve()

    # 1. 遍历第一级子目录
    for first_level_dir in [x for x in source_path.iterdir() if x.is_dir()]:
        
        # 避开逻辑：如果这个一级目录本身就是输出目录，直接跳过
        if first_level_dir == output_path:
            continue

        folder_name = first_level_dir.name
        print(f"正在处理一级目录: {folder_name}")

        # 2. 准备目标文件夹路径
        target_dir = output_path / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 3. 递归查找文件
        for file_path in first_level_dir.rglob("*"):
            
            # --- 关键改动：防止递归读取到输出目录 ---
            # 如果当前文件路径是以输出路径开头的，说明它在 data_vanilla 内部，跳过它
            try:
                if file_path.resolve().is_relative_to(output_path):
                    continue
            except ValueError:
                # 如果 file_path 不在 output_path 内，会抛出 ValueError，这是正常情况
                pass

            # --- 扩展位置：在这里添加你的匹配逻辑 ---
            supported_extensions = [".wav"] 
            
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                destination = target_dir / file_path.name
                
                try:
                    shutil.copy2(file_path, destination)
                    print(f"  已复制: {file_path.name} -> {target_dir}")
                except Exception as e:
                    print(f"  复制失败 {file_path.name}: {e}")

def main():
    # --- 路径配置区 ---
    CONFIG = {
        "source_dir": "data/data_download",        
        "output_dir": "data/data_vanilla" # 明确避开这个输出目录
    }

    process_audio_files(CONFIG["source_dir"], CONFIG["output_dir"])
    print("\n任务完成！")

if __name__ == "__main__":
    main()