import os
import argparse
import sys
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from reasoning import predict_speaker_for_file


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


def evaluate_file(audio_path, model1, model2, labels_dir, device):
    # Suppress verbose output from predict_speaker_for_file
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        results, _ = predict_speaker_for_file(audio_path, model1, model2, device)
    finally:
        sys.stdout = old_stdout
        devnull.close()

    if not results:
        return [], [], [], []

    base = os.path.splitext(os.path.basename(audio_path))[0]
    label_file = os.path.join(labels_dir, base + '.txt')
    if not os.path.exists(label_file):
        return [], [], [], []

    gt_intervals = load_ground_truth(label_file)

    y_true_vad, y_pred_vad = [], []
    y_true_sid, y_pred_sid = [], []

    for seg in results:
        start_sample = seg['start']
        end_sample = seg['end']
        pred_label = seg['pred_class']
        center = (start_sample + end_sample) // 2
        true_label = assign_gt_label(gt_intervals, center)

        # VAD
        y_true_vad.append(0 if true_label == 0 else 1)
        y_pred_vad.append(0 if pred_label == 0 else 1)
        # SID
        if true_label != 0:
            y_true_sid.append(true_label)
            y_pred_sid.append(pred_label)

    return y_true_vad, y_pred_vad, y_true_sid, y_pred_sid


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate VAD+SID over a directory of .wav files.")
    parser.add_argument('--audio_dir', default="./evaluate_wav", help="Directory containing .wav files to evaluate")
    parser.add_argument('--model1', default="./model/voice_detector_trained_968_967.pth",
                        help="Path to stage1 VAD model weights")
    parser.add_argument('--model2', default="./model/speaker_classifier_trained_948_943_934.pth",
                        help="Path to stage2 SID model weights")
    parser.add_argument('--labels_dir', default='./output_label',
                        help="Directory containing ground truth label .txt files")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Aggregate lists
    agg_true_vad, agg_pred_vad = [], []
    agg_true_sid, agg_pred_sid = [], []

    # Iterate over wav files
    for fname in sorted(os.listdir(args.audio_dir)):
        if not fname.lower().endswith('.wav'):
            continue
        audio_path = os.path.join(args.audio_dir, fname)
        tv, pv, ts, ps = evaluate_file(audio_path, args.model1, args.model2, args.labels_dir, device)
        agg_true_vad.extend(tv)
        agg_pred_vad.extend(pv)
        agg_true_sid.extend(ts)
        agg_pred_sid.extend(ps)

    # Overall VAD report
    print("\n===== Overall VAD Evaluation =====")
    print(classification_report(agg_true_vad, agg_pred_vad,
                                labels=[0,1],
                                target_names=["no voice","with voice"],
                                zero_division=0))
    prec_vad = precision_score(agg_true_vad, agg_pred_vad, average='binary', pos_label=1, zero_division=0)
    rec_vad = recall_score(agg_true_vad, agg_pred_vad, average='binary', pos_label=1, zero_division=0)
    f1_vad = f1_score(agg_true_vad, agg_pred_vad, average='binary', pos_label=1, zero_division=0)
    print(f"Aggregated VAD Precision: {prec_vad:.4f} | Recall: {rec_vad:.4f} | F1-score: {f1_vad:.4f}")

    # Overall SID report
    print("\n===== Overall SID Evaluation =====")
    if agg_true_sid:
        report = classification_report(
            agg_true_sid, agg_pred_sid,
            labels=[1,2,3],
            target_names=["XiaoXin","XiaoYuan","others"],
            zero_division=0
        )
        # remove macro avg line
        for line in report.splitlines():
            if "macro avg" not in line:
                print(line)
        prec_sid = precision_score(agg_true_sid, agg_pred_sid, average='micro', zero_division=0)
        rec_sid = recall_score(agg_true_sid, agg_pred_sid, average='micro', zero_division=0)
        f1_sid = f1_score(agg_true_sid, agg_pred_sid, average='micro', zero_division=0)
        print(f"Aggregated SID Precision (micro): {prec_sid:.4f} | Recall: {rec_sid:.4f} | F1-score: {f1_sid:.4f}")
    else:
        print("No voice segments found across all files; SID evaluation skipped.")


if __name__ == '__main__':
    main()
