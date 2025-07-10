import os
import math
import torchaudio
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# --- 参数设置 ---
SAMPLING_RATE = 8000
FRAME_T = 0.25
OVERLAP_THRESHOLD = 0.6

FORCE_NO_SPEECH_KEYWORDS = ["blank", "cat", "dog", "environment", "music", "noise","nature"]
# 特定说话人A的关键字和标签
SPEAKER_A_KEYWORD = "XiaoXin"
SPEAKER_A_LABEL = 1
# 特定说话人B的关键字和标签
SPEAKER_B_KEYWORD = "XiaoYuan"
SPEAKER_B_LABEL = 2
# 默认其他说话人标签
OTHER_SPEAKER_LABEL = 3
# 默认无人声标签
NO_SPEECH_LABEL = 0

# --- 初始化 Silero VAD 模型 ---
print("正在初始化 Silero VAD 模型...")
model = load_silero_vad()
print("VAD 模型初始化完成。")

# --- 文件夹路径设置 ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
audio_folder = os.path.join(current_dir, 'evaluate_wav')
label_folder = os.path.join(current_dir, 'output_label')
os.makedirs(label_folder, exist_ok=True)

# --- 开始处理 ---
audio_files = [f for f in os.listdir(audio_folder) if f.lower().endswith('.wav')]
print(f"\n发现 {len(audio_files)} 个音频文件，开始批量处理...\n")

# 遍历所有音频文件
for idx, audio_file in enumerate(audio_files):
    print(f"【文件 {idx + 1}/{len(audio_files)}】正在处理：{audio_file}")
    audio_path = os.path.join(audio_folder, audio_file)
    file_base = os.path.splitext(audio_file)[0]

    try:
        wav, sr = torchaudio.load(audio_path)
        if sr != SAMPLING_RATE:
            wav = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(wav)
        total_samples = wav.size(-1)
        duration = total_samples / SAMPLING_RATE
        # 向下取整，舍弃末尾不完整片段
        num_segments = int(duration / FRAME_T)
        segment_samples = int(FRAME_T * SAMPLING_RATE)

        if num_segments == 0:
            print(f"⚠️ 音频时长不足 {FRAME_T} 秒，跳过。")
            print("-" * 60)
            continue

        labels = []

        # 规则 1: 检查是否为强制非人声文件
        is_forced_no_speech = any(keyword in file_base.lower() for keyword in FORCE_NO_SPEECH_KEYWORDS)

        if is_forced_no_speech:
            print(f"  规则匹配：文件名包含强制非人声关键字。所有帧标记为 {NO_SPEECH_LABEL} (无人声)。")
            for i in range(num_segments):
                start = i * segment_samples
                end = start + segment_samples
                labels.append((start, end, NO_SPEECH_LABEL))

        else:
            # 规则 2: 对于非强制非人声文件，使用 VAD 进行判断
            print("  规则匹配：使用 Silero VAD 进行语音检测。")
            audio = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
            speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=SAMPLING_RATE)

            # 根据文件名确定人声标签
            if SPEAKER_A_KEYWORD.lower() in file_base.lower():
                speech_label_to_use = SPEAKER_A_LABEL
                print(f"  说话人识别：文件名包含 '{SPEAKER_A_KEYWORD}'，人声标签设为 {speech_label_to_use}。")
            elif SPEAKER_B_KEYWORD.lower() in file_base.lower():
                speech_label_to_use = SPEAKER_B_LABEL
                print(f"  说话人识别：文件名包含 '{SPEAKER_B_KEYWORD}'，人声标签设为 {speech_label_to_use}。")
            else:
                speech_label_to_use = OTHER_SPEAKER_LABEL
                print(f"  说话人识别：文件名不匹配特定说话人，人声标签设为 {speech_label_to_use} (其他人说话)。")

            for i in range(num_segments):
                start = i * segment_samples
                end = start + segment_samples
                is_speech = False

                # 检查该片段是否与VAD检测到的任一语音段有足够重叠
                for t in speech_timestamps:
                    overlap_start = max(start, t['start'])
                    overlap_end = min(end, t['end'])
                    overlap_duration = overlap_end - overlap_start

                    if overlap_duration >= segment_samples * OVERLAP_THRESHOLD:
                        is_speech = True
                        break

                # 根据VAD结果分配标签
                speaker_type = speech_label_to_use if is_speech else NO_SPEECH_LABEL
                labels.append((start, end, speaker_type))

        label_path = os.path.join(label_folder, f"{file_base}.txt")
        with open(label_path, 'w') as f:
            for start, end, speaker_type in labels:
                f.write(f"{start} {end} {speaker_type}\n")

        print(f"✅ 已保存标签到 {label_path}")
        print(f"  总片段数: {num_segments}")
        print("-" * 60)

    except Exception as e:
        print(f"❌ 处理失败：{audio_file}，错误：{str(e)}")