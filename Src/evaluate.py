import os
import argparse
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from reasoning import predict_speaker_for_file, FS, OUTPUT_WINDOW_T

def load_ground_truth(label_path):
    gt = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start, end, label = map(int, parts)
            gt.append((start, end, label))
    return gt


def assign_gt_label(gt_intervals, center_sample):
    for start, end, label in gt_intervals:
        if start <= center_sample < end:
            return label
    return 0

def evaluate(audio_path, model_weights1, model_weights2, labels_dir, device):
    results, total_duration = predict_speaker_for_file(audio_path,
                                                       model_weights1,
                                                       model_weights2,
                                                       device)
    if not results:
        print("No inference results to evaluate.")
        return

    # Load ground truth file
    base = os.path.splitext(os.path.basename(audio_path))[0]
    label_file = os.path.join(labels_dir, base + '.txt')
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")

    gt_intervals = load_ground_truth(label_file)

    y_true_sid = []  # ground truth for speaker identity
    y_pred_sid = []  # predicted speaker identity
    y_true_vad = []  # ground truth for VAD (0 or 1)
    y_pred_vad = []  # predicted VAD (0 or 1)

    for segment in results:
        start_sample = segment['start']
        end_sample = segment['end']
        pred_label = segment['pred_class']

        # 聚类后每段是0.5秒，取中心点用于比对
        center = (start_sample + end_sample) // 2
        true_label = assign_gt_label(gt_intervals, center)

        # ==== VAD 评估 ====
        gt_vad = 0 if true_label == 0 else 1  # 0: silence, 1: voice
        pred_vad = 0 if pred_label == 0 else 1

        y_true_vad.append(gt_vad)
        y_pred_vad.append(pred_vad)

        # ==== SID 评估 ====
        if gt_vad == 1:  # 仅对有声帧评估说话人
            y_true_sid.append(true_label)
            y_pred_sid.append(pred_label)

     # ==== 输出文件名和评估标题 ====
    filename = os.path.basename(audio_path)
    red_bold = "\033[1;31m"  # 红色加粗
    reset = "\033[0m"  # 重置样式
    print(f"\n{red_bold}{'=' * 60}")
    print(f"文件 {filename} 推理结果指标评估".center(60))
    print(f"{'=' * 60}{reset}")

    # ==== 输出 VAD 评估 ====
    print("\n===== VAD Evaluation =====")
    print(classification_report(y_true_vad, y_pred_vad,
                                labels=[0, 1],
                                target_names=["no voice", "with voice"],
                                zero_division=0))
    precision_vad = precision_score(y_true_vad, y_pred_vad, average='binary', pos_label=1, zero_division=0)
    recall_vad = recall_score(y_true_vad, y_pred_vad, average='binary', pos_label=1, zero_division=0)
    f1_vad = f1_score(y_true_vad, y_pred_vad, average='binary', pos_label=1, zero_division=0)
    print(f"VAD Precision: {precision_vad:.4f} | Recall: {recall_vad:.4f} | F1-score: {f1_vad:.4f}")

    # ==== 输出 SID 评估 ====
    print("\n===== SID Evaluation =====")
    if y_true_sid:
        report_str = classification_report(
            y_true_sid, y_pred_sid,
            labels=[1, 2, 3],
            target_names=["XiaoXin", "XiaoYuan", "others"],
            zero_division=0
        )

        # 逐行输出，跳过包含 "macro avg" 的行
        for line in report_str.splitlines():
            if "macro avg" not in line:
                print(line)
        precision_sid = precision_score(y_true_sid, y_pred_sid, average='micro', zero_division=0)
        recall_sid = recall_score(y_true_sid, y_pred_sid, average='micro', zero_division=0)
        f1_sid = f1_score(y_true_sid, y_pred_sid, average='micro', zero_division=0)
    #     print(f"SID Micro Precision: {precision_sid:.4f} | Recall: {recall_sid:.4f} | F1-score: {f1_sid:.4f}")
    # else:
    #     print("没有检测到任何非静音帧，无法评估说话人识别准确率。")

def main():
    parser = argparse.ArgumentParser(description="Evaluate VAD+SID inference against ground truth labels.")
    parser.add_argument('--audio', default="./evaluate_wav/cyw.wav", help="Path to audio file")
    parser.add_argument('--model1', default="./model/voice_detector_trained_967_968.pth",
                        help="Path to stage1 VAD model weights")
    parser.add_argument('--model2', default="./model/speaker_classifier_trained_948_943_934.pth",
                        help="Path to stage2 SID model weights")
    parser.add_argument('--labels_dir', default='./output_label',
                        help="Directory containing ground truth label .txt files")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate(args.audio, args.model1, args.model2, args.labels_dir, device)


if __name__ == '__main__':
    main()
