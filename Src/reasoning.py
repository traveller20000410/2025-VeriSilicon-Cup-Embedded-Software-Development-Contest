# !/usr/bin/python
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from model import CNN
from collections import defaultdict
import time

# --- 推理配置参数 ---
MODEL_WEIGHTS_PATH1 = "./model/voice_detector_trained_972_965.pth"
MODEL_WEIGHTS_PATH2 = "./model/speaker_classifier_trained_948_943_934.pth"
# AUDIO_FILE_PATH = "./sound_set/XiaoYuan_0.wav"
# AUDIO_FILE_PATH = "./evaluate_wav/XiaoYuan_0.wav"
# AUDIO_FILE_PATH = "./evaluate_wav/D32_979.wav"
# AUDIO_FILE_PATH = "./Preliminary_Vocal/ID2_4.wav"
AUDIO_FILE_PATH = "./evaluate_wav/lxq.wav"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 音频和特征参数 ---
FS = 8000
INPUT_FRAME_T = 0.25
OUTPUT_WINDOW_T = 0.5
FRAME_STEP = 0.25

N_FFT = 512
HOP_LENGTH_MEL = 185
N_MELS = 32

mel_spectrogram_transform_inference = None
amplitude_to_db_transform_inference = None

# 类别映射
CLASS_MAPPING = {0: "无人声",1: "说话人ID：小芯",2: "说话人ID：小原",3: "说话人ID：其他说话人"}
CLASS_MAPPING_STAGE1 = {0: "无人声", 1: "有人声"}
CLASS_MAPPING_STAGE2 = {0: "说话人ID：小芯", 1: "说话人ID：小原", 2: "说话人ID：其他说话人"}

def initialize_inference_transforms():
    global mel_spectrogram_transform_inference, amplitude_to_db_transform_inference
    mel_spectrogram_transform_inference = T.MelSpectrogram(
        sample_rate=FS, n_fft=N_FFT, win_length=None, hop_length=HOP_LENGTH_MEL,
        n_mels=N_MELS, power=2.0,center=False)
    amplitude_to_db_transform_inference = T.AmplitudeToDB(stype='power', top_db=80)


def feature_transform_for_inference(audio_chunk_tensor: torch.Tensor):
    if mel_spectrogram_transform_inference is None or amplitude_to_db_transform_inference is None:
        raise RuntimeError("Transforms not initialized. Call initialize_inference_transforms() first.")

    if audio_chunk_tensor.dim() != 1:
        raise ValueError(
            f"Expected 1D tensor, got {audio_chunk_tensor.dim()}D tensor with shape {audio_chunk_tensor.shape}")

    win_samples = int(FS * INPUT_FRAME_T)
    if audio_chunk_tensor.size(0) < win_samples:
            padding = win_samples - audio_chunk_tensor.size(0)
            audio_chunk_tensor = torch.nn.functional.pad(audio_chunk_tensor, (0, padding))

    # 计算梅尔频谱图
    mel_spec = mel_spectrogram_transform_inference(audio_chunk_tensor)
    mel_spec_db = amplitude_to_db_transform_inference(mel_spec)

    # 归一化
    mean = mel_spec_db.mean()
    std = mel_spec_db.std()
    if std > 1e-5:
        mel_spec_db_normalized = (mel_spec_db - mean) / std
    else:
        mel_spec_db_normalized = mel_spec_db - mean

    return mel_spec_db_normalized.unsqueeze(0)


def segment_audio_for_inference(waveform: torch.Tensor, fs: int = FS,
                                frame_t: float = INPUT_FRAME_T,
                                frame_step: float = FRAME_STEP):
    total_samples = waveform.size(-1)
    win_samples = int(frame_t * fs)
    step_samples = int(frame_step * fs)

    audio_segments = []
    segment_positions = []
    segment_end_positions = []

    for s_sample in range(0, total_samples - win_samples + 1, step_samples):
        e_sample = s_sample + win_samples
        audio_chunk_tensor = waveform[0, s_sample:e_sample].float()

        audio_segments.append(audio_chunk_tensor)
        segment_positions.append(s_sample)
        segment_end_positions.append(e_sample)

    return audio_segments, segment_positions, segment_end_positions

def aggregate_results_2s(predictions, positions, end_positions,
                        output_window_t=OUTPUT_WINDOW_T, sample_rate=FS):
    if not predictions:
        return []

    frames_per_window = int(output_window_t / FRAME_STEP)
    results = []
    window_id = 0

    while window_id * frames_per_window < len(predictions):
        start_idx = window_id * frames_per_window
        end_idx = min((window_id + 1) * frames_per_window, len(predictions))
        window_preds = predictions[start_idx:end_idx]

        # 统计每个类别的总置信度和计数
        conf_sums = defaultdict(float)
        counts = defaultdict(int)
        for cls, conf in window_preds:
            conf_sums[cls] += conf
            counts[cls] += 1

        # 计算平均置信度
        avg_confs = {cls: conf_sums[cls] / counts[cls] for cls in counts}

        # 选出平均置信度最高的类别
        final_class, final_conf = max(avg_confs.items(), key=lambda x: x[1])

        # 确定窗口在原始音频中的时间
        win_start_sample = positions[start_idx]
        win_end_sample   = end_positions[end_idx - 1]

        results.append({
            'start': win_start_sample,
            'end':   win_end_sample,
            'pred_class':    final_class,
            'pred_label':    CLASS_MAPPING[final_class],
            'confidence':    final_conf,
            'class_distribution': dict(counts),
            'frame_count':   end_idx - start_idx
        })

        window_id += 1

    return results



def predict_speaker_for_file(audio_path: str, model_weights_path1: str, model_weights_path2: str, device: torch.device):
    initialize_inference_transforms()

    model1 = CNN(num_classes=2).to(device)
    try:
        model1.load_state_dict(torch.load(model_weights_path1, map_location=device, weights_only=True))
        print(f"加载第一阶段模型权重成功: {model_weights_path1}")
    except FileNotFoundError:
        print(f"错误: 第一阶段模型权重文件未找到于 {model_weights_path1}")
        return []
    except Exception as e:
        print(f"错误: 加载第一阶段模型权重失败: {e}")
        return []

    model2 = CNN(num_classes=3).to(device)
    try:
        model2.load_state_dict(torch.load(model_weights_path2, map_location=device, weights_only=True))
        print(f"加载第二阶段模型权重成功: {model_weights_path2}")
    except FileNotFoundError:
        print(f"错误: 第二阶段模型权重文件未找到于 {model_weights_path2}")
        return []
    except Exception as e:
        print(f"错误: 加载第二阶段模型权重失败: {e}")
        return []

    model1.eval()
    model2.eval()

    try:
        waveform, sr = torchaudio.load(audio_path)
        duration = waveform.size(1) / sr
        print(f"音频加载成功 | 时长: {duration:.2f}秒 | 采样率: {sr}Hz")
    except FileNotFoundError:
        print(f"错误: 音频文件未找到于 {audio_path}")
        return []
    except Exception as e:
        print(f"错误: 加载音频文件失败: {e}")
        return []

    if sr != FS:
        print(f"音频采样率{sr}Hz，重采样为{FS}Hz")
        resampler = torchaudio.transforms.Resample(sr, FS)
        waveform = resampler(waveform)

    if waveform.size(0) > 1:
        print("立体声音频已混合为单声道")
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_segments, segment_positions, segment_end_positions = segment_audio_for_inference(waveform)

    if not audio_segments:
        print("未能从音频文件中切分出任何片段。")
        return []

    total_frames = len(audio_segments)
    total_audio_duration = waveform.size(1) / FS
    print(f"共切分出 {total_frames} 个音频片段 | "
          f"每帧时长: {INPUT_FRAME_T:.2f}秒 | "
          f"总处理时长: {total_frames * INPUT_FRAME_T:.2f}秒 | "
          f"实际音频时长: {total_audio_duration:.2f}秒")

    predictions = []
    batch_size = 32
    total_batches = (total_frames + batch_size - 1) // batch_size

    print(f"开始两阶段模型推理，批次大小: {batch_size}，总批次: {total_batches}")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx in range(0, total_frames, batch_size):
            batch_segments = audio_segments[batch_idx:batch_idx + batch_size]
            batch_features = []

            for segment in batch_segments:
                feature = feature_transform_for_inference(segment)
                batch_features.append(feature)

            if batch_features:
                batch_tensor = torch.cat(batch_features, dim=0).unsqueeze(1).to(device)
                print(f"Batch tensor shape: {batch_tensor.shape}")  # 调试输出

                logits1 = model1(batch_tensor)
                probabilities1 = torch.softmax(logits1, dim=1)
                conf1, preds1 = torch.max(probabilities1, dim=1)

                voice_indices = (preds1 == 1).nonzero(as_tuple=True)[0]
                if len(voice_indices) > 0:
                    voice_features = batch_tensor[voice_indices]
                    logits2 = model2(voice_features)
                    probabilities2 = torch.softmax(logits2, dim=1)
                    conf2, preds2 = torch.max(probabilities2, dim=1)
                else:
                    conf2 = torch.tensor([]).to(device)
                    preds2 = torch.tensor([]).to(device)

                for i in range(len(batch_segments)):
                    if i in voice_indices:
                        pred_id = preds2[voice_indices == i].item() + 1
                        conf = conf2[voice_indices == i].item()
                    else:
                        pred_id = 0
                        conf = conf1[i].item() if preds1[i].item() == 0 else 1.0
                    predictions.append((pred_id, conf))

            if (batch_idx // batch_size) % 10 == 0 or batch_idx + batch_size >= total_frames:
                elapsed = time.time() - start_time
                processed = min(batch_idx + batch_size, total_frames)
                print(f"\r处理进度: {processed}/{total_frames} 帧 "
                      f"({100. * processed / total_frames:.1f}%) | "
                      f"耗时: {elapsed:.1f}秒 | "
                      f"预计剩余: {(elapsed / (processed + 1)) * (total_frames - processed):.1f}秒", end='')

    elapsed_total = time.time() - start_time
    print(f"\n推理完成 | 总耗时: {elapsed_total:.2f}秒 | "
          f"平均速度: {total_frames / elapsed_total:.2f}帧/秒")

    print(f"聚合每{OUTPUT_WINDOW_T}秒的结果...")
    aggregated_results = aggregate_results_2s(
        predictions,
        segment_positions,
        segment_end_positions,
        output_window_t=OUTPUT_WINDOW_T,
        sample_rate=FS
    )

    return aggregated_results, total_audio_duration


def display_results(results, total_duration):
    print(f"\n{'时间段':<15} | {'持续时间':<8} | {'预测状态':<12} | {'置信度':<7} | {'类别分布'}")
    print("-" * 70)

    for result in results:
        start_sec = result['start'] / FS
        end_sec = result['end'] / FS
        duration = end_sec - start_sec
        dist_str = ", ".join(f"{CLASS_MAPPING[k]}:{v}" for k, v in sorted(result['class_distribution'].items()))
        print(f"{start_sec:.2f}-{end_sec:.2f}s | {duration:.2f}s   | "
              f"{result['pred_label']:<12} | {result['confidence']:.4f} | {dist_str}")

    print("\n整体统计:")
    total_duration_by_class = {label: 0.0 for label in CLASS_MAPPING.values()}
    for result in results:
        duration = (result['end'] - result['start']) / FS
        total_duration_by_class[result['pred_label']] += duration

    for label, duration in total_duration_by_class.items():
        percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
        print(f"{label}: {percentage:.2f}% ({duration:.2f}秒)")

if __name__ == "__main__":
    script_start_time = time.time()
    if not Path(AUDIO_FILE_PATH).exists():
        print(f"错误: 测试音频文件 '{AUDIO_FILE_PATH}' 不存在。请修改 AUDIO_FILE_PATH。")
        exit()

    print(f"正在对音频文件进行两阶段说话人识别: {AUDIO_FILE_PATH}")
    print(f"使用第一阶段模型权重: {MODEL_WEIGHTS_PATH1}")
    print(f"使用第二阶段模型权重: {MODEL_WEIGHTS_PATH2}")
    print(f"使用设备: {DEVICE}")
    print(f"输入帧长: {INPUT_FRAME_T:.1f}秒 | 输出间隔: {OUTPUT_WINDOW_T:.1f}秒\n")

    results, total_duration = predict_speaker_for_file(AUDIO_FILE_PATH, MODEL_WEIGHTS_PATH1, MODEL_WEIGHTS_PATH2,DEVICE)

    if results:
        display_results(results, total_duration)
    else:
        print("未检测到任何有效结果。")
    script_total_time = time.time() - script_start_time
    print(f"\n脚本整体执行时间（包括导入和打印）：{script_total_time:.2f}秒")