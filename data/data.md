unzip /data/yhy/code/DabuAudio/data/data_download/新建文件夹.zip -d /data/yhy/code/DabuAudio/data/data_download/hxx

# 数据预处理
0. 原始音频需要归一化，比如变成16k采样率，单通道的wav文件，归一化后放在data/data_norm目录下
1. 获取原始音频后，要选择降噪吗，降噪完毕后放在data/data_denoise目录下
2. 对音频进行vad+asr吗，分帧后放在data/data_vad_asr目录下,例如data/data_vad_asr/hxx/audio1.wav // data/data_vad_asr/hxx/audio1.jsonl作为asr
3. 也可以选择不降噪，直接vadasr，vad片段作为jsonl的片段，
