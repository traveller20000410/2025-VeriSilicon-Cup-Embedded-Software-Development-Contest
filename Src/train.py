#!/usr/bin/python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from model import CNN
import time
import multiprocessing
from collections import defaultdict
from sklearn.metrics import f1_score

# --- 超参数 ---
LEARNING_RATE = 8e-4
BATCH_SIZE = 256
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = Path("./model/speaker_classifier_trained.pth")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- 早停参数 ---
PATIENCE = 50
BEST_VAL_ACC = 0.0
BEST_EPOCH = 0
NO_IMPROVEMENT_COUNT = 0

# --- 数据目录 ---
BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "sound_set"
LABEL_DIR = BASE_DIR / "output_label"

# --- 音频参数 ---
FS = 8000
FRAME_T = 0.25
FRAME_STEP = 0.25

# --- 梅尔频谱图参数 ---
N_FFT = 512
HOP_LENGTH_MEL = 185  #240，210，185
N_MELS = 32

mel_spectrogram_transform_global = None
amplitude_to_db_transform_global = None

CLASS_MAPPING = {0: "无人声",1: "A说话",2: "B说话",3: "其他人说话"}

def initialize_audio_transforms():
    global mel_spectrogram_transform_global, amplitude_to_db_transform_global
    mel_spectrogram_transform_global = T.MelSpectrogram(
        sample_rate=FS,
        n_fft=N_FFT,
        win_length=None,
        hop_length=HOP_LENGTH_MEL,
        n_mels=N_MELS,
        power=2.0,
        center=False
    )
    amplitude_to_db_transform_global = T.AmplitudeToDB(stype='power', top_db=80)


def load_labels(label_path: Path):
    intervals = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start, end = int(parts[0]), int(parts[1])
                speaker_type = int(parts[2])
                intervals.append((start, end, speaker_type))
    return intervals

def segment_audio(waveform: torch.Tensor, fs: int = FS):
    total = waveform.size(-1)
    win = int(FRAME_T * fs)
    step = int(FRAME_STEP * fs)
    chunks = []
    positions = []  # 记录片段的起始位置

    for s in range(0, total - win + 1, step):
        chunk = waveform[0, s:s + win].float()
        chunks.append((s, chunk))
        positions.append(s)

    return chunks, positions

def label_chunks(chunks, positions, intervals, energy_threshold_db=-40):
    labels = []
    win_len_samples = int(FRAME_T * FS)

    for pos_idx, (s, audio_chunk) in enumerate(chunks):
        segment_start = positions[pos_idx]
        segment_end = segment_start + win_len_samples

        segment_label = 0

        power = torch.mean(audio_chunk ** 2)
        energy_db = 10 * torch.log10(power + 1e-9)

        for st, ed, speaker_type in intervals:
            overlap_start = max(segment_start, st)
            overlap_end = min(segment_end, ed)
            overlap_duration = overlap_end - overlap_start

            # 如果有足够重叠，使用说话人类型标签
            if overlap_duration > 0 and overlap_duration >= win_len_samples * 0.6:
                segment_label = speaker_type
                break

        labels.append(segment_label)

    return labels


class SpeakerDataset(Dataset):
    def __init__(self, audio_tensor_list, labels, transform_function):
        self.audio_tensor_list = audio_tensor_list
        self.labels = labels
        self.transform = transform_function

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_tensor = self.audio_tensor_list[idx]
        lb = self.labels[idx]
        feat = self.transform(audio_tensor)
        return feat, torch.tensor(lb, dtype=torch.long)

def prepare_dataloaders(num_workers=0):
    all_audio_tensors = []
    all_labels_stage1 = []
    all_labels_stage2 = []
    label_stats_stage1 = defaultdict(int)
    label_stats_stage2 = defaultdict(int)

    # 遍历所有音频文件
    for wav_file in AUDIO_DIR.glob("*.wav"):
        base = wav_file.stem
        lab_file = LABEL_DIR / f"{base}.txt"

        if not lab_file.exists():
            print(f"警告: 未找到 {wav_file} 对应的标签文件 {lab_file}。跳过。")
            continue

        try:
            wav, sr = torchaudio.load(wav_file)
        except Exception as e:
            print(f"加载 {wav_file} 出错: {e}。跳过。")
            continue

        if sr != FS:
            wav = torchaudio.transforms.Resample(sr, FS)(wav)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        intervals = load_labels(lab_file)
        chunk_tuples, positions = segment_audio(wav, FS)

        if not chunk_tuples:
            print(f"警告: {wav_file} 未能生成任何音频片段。跳过。")
            continue

        labels_for_chunks = label_chunks(chunk_tuples, positions, intervals)

        for idx, (_, audio_chunk) in enumerate(chunk_tuples):
            all_audio_tensors.append(audio_chunk)
            original_label = labels_for_chunks[idx]
            # 第一阶段标签：无人声(0) vs 有语音(1)
            stage1_label = 0 if original_label == 0 else 1
            all_labels_stage1.append(stage1_label)
            label_stats_stage1[stage1_label] += 1

            if original_label != 0:
                stage2_label = original_label - 1
                all_labels_stage2.append(stage2_label)
                label_stats_stage2[stage2_label] += 1
            else:
                all_labels_stage2.append(-1)

    if not all_audio_tensors:
        raise ValueError("未能加载任何数据。请检查音频路径、标签文件和分段长度。")

    print("第一阶段标签分布（无人声 vs 有语音）：")
    total_samples_stage1 = len(all_labels_stage1)
    for label_id, count in sorted(label_stats_stage1.items()):
        label_name = "无人声" if label_id == 0 else "有语音"
        percentage = (count / total_samples_stage1) * 100
        print(f"  {label_name}: {count} 样本 ({percentage:.2f}%)")

    print("第二阶段标签分布（A vs B vs 其他人）：")
    total_samples_stage2 = sum(label_stats_stage2.values())
    for label_id, count in sorted(label_stats_stage2.items()):
        label_name = {0: "A说话", 1: "B说话", 2: "其他人说话"}.get(label_id, f"未知标签({label_id})")
        percentage = (count / total_samples_stage2) * 100
        print(f"  {label_name}: {count} 样本 ({percentage:.2f}%)")

    # 划分数据集
    # 第一阶段：所有数据
    stratify_labels_stage1 = all_labels_stage1
    x_tr1, x_val1, y_tr1, y_val1 = train_test_split(
        all_audio_tensors, all_labels_stage1, test_size=0.2,
        random_state=42, stratify=stratify_labels_stage1
    )
    ds_tr1 = SpeakerDataset(x_tr1, y_tr1, mel_transform)
    ds_val1 = SpeakerDataset(x_val1, y_val1, mel_transform)
    # num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() - 2)
    dl_tr1 = DataLoader(ds_tr1, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    dl_val1 = DataLoader(ds_val1, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    # 第二阶段：仅包含有语音的数据
    valid_indices = [i for i, lb in enumerate(all_labels_stage2) if lb != -1]
    audio_tensors_stage2 = [all_audio_tensors[i] for i in valid_indices]
    labels_stage2 = [all_labels_stage2[i] for i in valid_indices]
    stratify_labels_stage2 = labels_stage2
    x_tr2, x_val2, y_tr2, y_val2 = train_test_split(
        audio_tensors_stage2, labels_stage2, test_size=0.2,
        random_state=42, stratify=stratify_labels_stage2
    )
    ds_tr2 = SpeakerDataset(x_tr2, y_tr2, mel_transform)
    ds_val2 = SpeakerDataset(x_val2, y_val2, mel_transform)
    dl_tr2 = DataLoader(ds_tr2, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    dl_val2 = DataLoader(ds_val2, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    return (dl_tr1, dl_val1, label_stats_stage1), (dl_tr2, dl_val2, label_stats_stage2)


def mel_transform(audio_chunk_tensor: torch.Tensor):
    if audio_chunk_tensor.size(0) < int(FS * 0.4):
        padding_size = max(0, int(FS * FRAME_T) - audio_chunk_tensor.size(0))
        audio_chunk_tensor = torch.nn.functional.pad(audio_chunk_tensor, (0, padding_size))

    # 计算梅尔频谱图
    mel_spec = mel_spectrogram_transform_global(audio_chunk_tensor)
    mel_spec_db = amplitude_to_db_transform_global(mel_spec)

    # 归一化
    mean = mel_spec_db.mean()
    std = mel_spec_db.std()
    if std > 1e-6:
        mel_spec_db_normalized = (mel_spec_db - mean) / std
    else:
        mel_spec_db_normalized = mel_spec_db - mean

    return mel_spec_db_normalized.unsqueeze(0)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            out = model(X)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * X.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            total_loss += loss.item() * X.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy, all_preds, all_targets

def calculate_class_weights(label_counts, num_classes, boost_factor=0.5):
    total_samples = sum(label_counts.values())
    class_weights = torch.ones(num_classes, dtype=torch.float32)

    for class_id in range(num_classes):
        class_count = label_counts.get(class_id, 0)
        if class_count == 0:
            class_weights[class_id] = 1.0
        else:
            class_weights[class_id] = total_samples / (num_classes * class_count)

    if num_classes == 2:
        class_weights[0] = 1.0
        class_weights[1] = 1.2
        class_weights /= class_weights.sum()
        return class_weights

    if num_classes == 3:
        class_weights[0] = 1.0
        class_weights[1] = 2.0
        class_weights[2] = 1.0
        class_weights /= class_weights.sum()
    return class_weights

# def calculate_class_weights(label_counts, num_classes, boost_factor=1.0):
#     total_samples = sum(label_counts.values())
#     class_weights = torch.ones(num_classes, dtype=torch.float32)
#
#     for class_id in range(num_classes):
#         class_count = label_counts.get(class_id, 0)
#         if class_count == 0:
#             # 如果某个类别没有样本，保持默认权重1.0
#             class_weights[class_id] = 1.0
#         else:
#             class_weights[class_id] = (total_samples / (num_classes * class_count)) * boost_factor
#
#     # 归一化权重，使权重总和等于类别数
#     class_weights /= class_weights.sum()
#     class_weights *= num_classes
#
#     return class_weights

if __name__ == "__main__":
    initialize_audio_transforms()
    multiprocessing.set_start_method('spawn', force=True)

    print("准备数据...")
    try:
        (train_loader1, val_loader1, label_stats1), (train_loader2, val_loader2, label_stats2) = prepare_dataloaders(num_workers=0)
        if len(train_loader1.dataset) == 0 or len(train_loader2.dataset) == 0:
            print("错误: 数据集为空。程序退出。")
            exit()

        # 检查输入特征形状
        sample_feat, _ = train_loader1.dataset[0]
        print(f"输入特征形状 (通道, 梅尔带数, 时间帧数): {sample_feat.shape}")

    except Exception as e:
        print(f"数据准备过程中发生错误: {e}")
        exit()

    # # 第一阶段模型：语音检测（二分类）
    # print("\n训练第一阶段模型（语音检测）...")
    # model1 = CNN(num_classes=2).to(DEVICE)
    # print("\n第一阶段模型结构:")
    # print(model1)
    #
    # class_weights1 = calculate_class_weights(label_stats1, num_classes=2, boost_factor=1.0)
    # print("\n第一阶段类别权重计算:")
    # print(f"类别0(无人声)权重: {class_weights1[0]:.4f}")
    # print(f"类别1(有语音)权重: {class_weights1[1]:.4f}")
    # class_weights1 = class_weights1.to(DEVICE)
    # criterion1 = nn.CrossEntropyLoss(weight=class_weights1,label_smoothing=0.1)
    # optimizer1 = optim.AdamW(model1.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # scaler1 = torch.amp.GradScaler('cuda')
    # scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer1, 'max', patience=PATIENCE // 5, factor=0.5)
    #
    # best_acc1 = 0.0
    # best_epoch1 = 0
    # no_improvement1 = 0
    # training_start_time = time.time()
    # MODEL_SAVE_PATH1 = Path("model_new/voice_detector_trained.pth")
    # MODEL_SAVE_PATH1.parent.mkdir(parents=True, exist_ok=True)
    #
    # for epoch in range(NUM_EPOCHS):
    #     epoch_start_time = time.time()
    #     train_loss, train_acc = train_one_epoch(
    #         model1, train_loader1, criterion1, optimizer1, DEVICE, scaler1)
    #     val_loss, val_acc, all_preds, all_targets = validate_one_epoch(
    #         model1, val_loader1, criterion1, DEVICE)
    #     epoch_end_time = time.time()
    #     duration = epoch_end_time - epoch_start_time
    #
    #     class_acc = {}
    #     for c in range(2):
    #         idx = np.array(all_targets) == c
    #         if sum(idx) > 0:
    #             class_acc["无人声" if c == 0 else "有语音"] = np.mean(np.array(all_preds)[idx] == c)
    #     f1 = f1_score(all_targets, all_preds, average='weighted')
    #
    #     print(f"第一阶段 Epoch {epoch + 1}/{NUM_EPOCHS} | 耗时: {duration:.2f}s")
    #     print(f"Weighted F1 Score: {f1:.4f}")
    #     print(f"  训练: 损失={train_loss:.4f}, 准确率={train_acc * 100:.2f}%")
    #     print(f"  验证: 损失={val_loss:.4f}, 准确率={val_acc * 100:.2f}%")
    #     print("  各类别验证准确率:")
    #     for cls, acc in class_acc.items():
    #         print(f"    {cls}: {acc * 100:.2f}%")
    #
    #     scheduler1.step(val_acc)
    #     current_lr = scheduler1.get_last_lr()
    #     print(f"Current LR: {current_lr}")
    #
    #     # 检查是否满足保存条件
    #     save_model = False
    #     if ("无人声" in class_acc and class_acc["无人声"] > 0.962 and
    #             "有语音" in class_acc and class_acc["有语音"] > 0.962):
    #         save_model = True
    #         no_voice_acc_int = int(class_acc["无人声"] * 1000)
    #         voice_acc_int = int(class_acc["有语音"] * 1000)
    #         model_filename = f"voice_detector_trained_{no_voice_acc_int}_{voice_acc_int}.pth"
    #         model_save_path = MODEL_SAVE_PATH1.parent / model_filename
    #     # 保存模型
    #     if save_model:
    #         torch.save(model1.state_dict(), model_save_path)
    #         print(f"----> 第一阶段模型已保存 (Epoch {epoch + 1}): {model_filename} <----")
    #
    #     # 早停逻辑（基于验证准确率）
    #     if val_acc > best_acc1:
    #         best_acc1 = val_acc
    #         best_epoch1 = epoch + 1
    #         no_improvement1 = 0
    #     else:
    #         no_improvement1 += 1
    #         print(
    #             f"  连续 {no_improvement1} 个 epochs 没有提升。当前最佳验证准确率: {best_acc1 * 100:.2f}% (Epoch {best_epoch1})")
    #     if no_improvement1 >= PATIENCE:
    #         print(f"\n第一阶段在 Epoch {epoch + 1} 触发早停。")
    #         break
    #
    # print(f"\n第一阶段训练完成，总耗时: {time.time() - training_start_time:.2f}s。")
    # print(f"最佳验证准确率: {best_acc1 * 100:.2f}% (在 Epoch {best_epoch1})")

    # 第二阶段模型：说话人分类（三分类）
    print("\n训练第二阶段模型（说话人分类）...")
    model2 = CNN(num_classes=3).to(DEVICE)
    print("\n第二阶段模型结构:")
    print(model2)

    class_weights2 = calculate_class_weights(label_stats2, num_classes=3, boost_factor=2.0)
    print("\n第二阶段类别权重计算:")
    print(f"类别0(A说话)权重: {class_weights2[0]:.4f}")
    print(f"类别1(B说话)权重: {class_weights2[1]:.4f}")
    print(f"类别2(其他人说话)权重: {class_weights2[2]:.4f}")
    class_weights2 = class_weights2.to(DEVICE)
    criterion2 = nn.CrossEntropyLoss(weight=class_weights2,label_smoothing=0.1)
    optimizer2 = optim.AdamW(model2.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scaler2 = torch.amp.GradScaler('cuda')
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, 'max', patience=PATIENCE // 4, factor=0.5)

    best_acc2 = 0.0
    best_epoch2 = 0
    no_improvement2 = 0
    training_start_time = time.time()
    MODEL_SAVE_PATH2 = Path("model_new/speaker_classifier_trained.pth")
    MODEL_SAVE_PATH2.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model2, train_loader2, criterion2, optimizer2, DEVICE, scaler2)
        val_loss, val_acc, all_preds, all_targets = validate_one_epoch(
            model2, val_loader2, criterion2, DEVICE)
        epoch_end_time = time.time()
        duration = epoch_end_time - epoch_start_time

        class_acc = {}
        for c in range(3):
            idx = np.array(all_targets) == c
            if sum(idx) > 0:
                class_acc[{0: "A说话", 1: "B说话", 2: "其他人说话"}[c]] = np.mean(np.array(all_preds)[idx] == c)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        print(f"第二阶段 Epoch {epoch + 1}/{NUM_EPOCHS} | 耗时: {duration:.2f}s")
        print(f"Weighted F1 Score: {f1:.4f}")
        print(f"  训练: 损失={train_loss:.4f}, 准确率={train_acc * 100:.2f}%")
        print(f"  验证: 损失={val_loss:.4f}, 准确率={val_acc * 100:.2f}%")
        print("  各类别验证准确率:")
        for cls, acc in class_acc.items():
            print(f"    {cls}: {acc * 100:.2f}%")

        save_model = False
        if ("A说话" in class_acc and class_acc["A说话"] > 0.94 and
                "B说话" in class_acc and class_acc["B说话"] > 0.93 and
                "其他人说话" in class_acc and class_acc["其他人说话"] > 0.94):
            save_model = True
            a_acc_int = int(class_acc["A说话"] * 1000)
            b_acc_int = int(class_acc["B说话"] * 1000)
            others_acc_int = int(class_acc["其他人说话"] * 1000)
            model_filename = f"speaker_classifier_trained_{a_acc_int}_{b_acc_int}_{others_acc_int}.pth"
            model_save_path = MODEL_SAVE_PATH2.parent / model_filename

        scheduler2.step(val_acc)
        current_lr = scheduler2.get_last_lr()
        print(f"Current LR: {current_lr}")

        # 保存模型
        if save_model:
            torch.save(model2.state_dict(), model_save_path)
            print(f"----> 第二阶段模型已保存 (Epoch {epoch + 1}): {model_filename} <----")

        if val_acc > best_acc2:
            best_acc2 = val_acc
            best_epoch2 = epoch + 1
            no_improvement2 = 0
        else:
            no_improvement2 += 1
            print(
                f"  连续 {no_improvement2} 个 epochs 没有提升。当前最佳验证准确率: {best_acc2 * 100:.2f}% (Epoch {best_epoch2})")
        if no_improvement2 >= PATIENCE:
            print(f"\n第二阶段在 Epoch {epoch + 1} 触发早停。")
            break

    print(f"\n第二阶段训练完成，总耗时: {time.time() - training_start_time:.2f}s。")
    print(f"最佳验证准确率: {best_acc2 * 100:.2f}% (在 Epoch {best_epoch2})")