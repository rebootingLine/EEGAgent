# ==========================
# Sleep EEGAgent 
# ==========================
import os
import json
import time
import numpy as np
from datetime import datetime
from main import EEGAgent 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import mne
import regex as re

# ==========================
# Basic Configuration
# ==========================
DATA_PATH = "./eval/sleep/data/file_test.npy"
EEG_DIR = "./eval/sleep/sleep-cassette"  # EDF文件所在文件夹
SAVE_DIR = "./eval/eval_logs/Sleep"
os.makedirs(SAVE_DIR, exist_ok=True)

# read test files
files = np.load(DATA_PATH)
print(f"Loaded {len(files)} test subjects.")

STAGE_LABELS = ["W", "N1", "N2", "N3", "R"]
LABEL_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

# ==========================
# experiment setup
# ==========================
SEGMENT_DURATION = 1800 
QUESTION_TEMPLATE = (
    "You are an expert in sleep EEG analysis. "
    "Analyze the EEG data between {start:.0f}s and {end:.0f}s. "
    "Determine the sleep stage (W, N1, N2, N3, R) for each 30-second epoch. "
    "There are exactly 60 epochs in this 30-minute segment. "
    "Return the results strictly as a valid Python list of tuples in this exact format:\n"
    "[(stage_name, start_time, end_time), (stage_name, start_time, end_time), ...]\n"
    "Each tuple must use plain English letters for the stage (no quotes), "
    "and numeric seconds for time (integers, not strings). Example:\n"
    "[(W, 0, 30), (N1, 30, 60), (N2, 60, 90), ...]\n"
    "Do not add explanations, extra text, or code comments — only return the list."
)
# ==========================

def extract_range(hyp_file, extend_wake=1800, random_pick=True):
    annots = mne.read_annotations(hyp_file)

    wake_onsets = [a['onset'] for a in annots if a['description'] == 'Sleep stage W']
    wake_durations = [a['duration'] for a in annots if a['description'] == 'Sleep stage W']

    if len(wake_onsets) < 2:
        sleep_start = 0.0
        sleep_end = annots[-1]['onset'] + annots[-1]['duration']
    else:
        gaps = [wake_onsets[i+1] - (wake_onsets[i] + wake_durations[i]) for i in range(len(wake_onsets)-1)]
        max_gap_idx = int(np.argmax(gaps))
        sleep_start = wake_onsets[max_gap_idx] + wake_durations[max_gap_idx]
        sleep_end = wake_onsets[max_gap_idx + 1]

    # 扩展并限制到 annotation 范围
    total_end = annots[-1]['onset'] + annots[-1]['duration']
    sleep_start = max(0.0, sleep_start - extend_wake)
    sleep_end = min(sleep_end + extend_wake, total_end)

    sleep_total = sleep_end - sleep_start
    if sleep_total <= SEGMENT_DURATION:
        start = sleep_start
        end = sleep_end
    else:
        if random_pick:
            max_offset = sleep_total - SEGMENT_DURATION
            n_slots = int(max_offset // 30)
            offset = 30 * random.randint(0, n_slots) if n_slots > 0 else 0
            start = sleep_start + offset
        else:
            start = sleep_start + (sleep_total - SEGMENT_DURATION) / 2.0
        end = start + SEGMENT_DURATION
        # 保险检查
        if end > total_end:
            end = total_end
            start = max(sleep_start, end - SEGMENT_DURATION)

    # debug 打印（运行一次看下输出）
    # print(f"[DEBUG] sleep_start={sleep_start:.1f}, sleep_end={sleep_end:.1f}, pick={start:.1f}-{end:.1f}, total_end={total_end:.1f}")
    return start, end


def load_ground_truth(hyp_file, start_time, end_time, epoch_len=30):
    annotations = mne.read_annotations(hyp_file)

    labels = []
    onset_arr = np.array([a['onset'] for a in annotations])
    offset_arr = np.array([a['onset'] + a['duration'] for a in annotations])
    desc_arr = np.array([a['description'] for a in annotations])

    # Divide from start_time to end_time every 30 seconds
    n_epochs = int(np.floor((end_time - start_time) / epoch_len))
    for i in range(n_epochs):
        epoch_start = start_time + i * epoch_len

        # find epoch in which annotation 
        idx = np.where((onset_arr <= epoch_start) & (epoch_start < offset_arr))[0]
        if len(idx) > 0:
            desc = desc_arr[idx[0]]
            if desc in LABEL_MAP:
                stage = ['W', 'N1', 'N2', 'N3', 'R'][LABEL_MAP[desc]]
            else:
                stage = 'W'
        else:
            stage = 'W'  # fallback

        labels.append(stage)

    return labels


# ==========================
# main loop
# ==========================
all_true, all_pred = [], []
results = []

for i, (psg_file, hyp_file) in enumerate(files):
    print(f"\n[{i+1}/{len(files)}] Evaluating {psg_file} ...")
    time.sleep(10)

    psg_path = os.path.join(EEG_DIR, psg_file)
    hyp_path = os.path.join(EEG_DIR, hyp_file)

    start_t, end_t = extract_range(hyp_path)
    question = QUESTION_TEMPLATE.format(start=start_t, end=end_t)

    agent = EEGAgent(
        config_path="config/config.json",
        file_name=psg_file,
        api_key="***",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    result = agent.run(question)
    response = result["response"]

    # Ground truth
    gt_segment = load_ground_truth(hyp_path, start_time=start_t, end_time=end_t)
    n_epochs = len(gt_segment)
    
    pattern = r"""
        \(\s*                   
        (?P<stage>W|N1|N2|N3|R)  
        \s*,\s*                  
        (?P<start>\d+(?:\.\d+)?) 
        \s*,\s*                 
        (?P<end>\d+(?:\.\d+)?)   
        \s*\)                     
    """
    matches = re.findall(pattern, response, flags=re.VERBOSE)
    pred_labels = [m[0] for m in matches if m[0] in STAGE_LABELS]

    if len(gt_segment) != SEGMENT_DURATION // 30 or len(pred_labels) != SEGMENT_DURATION // 30:
        print("\n[ERROR] Length mismatch detected!")
        print(f"Subject: {psg_file}")
        print(f"Ground truth length: {len(gt_segment)} | Prediction length: {len(pred_labels)}")
        if all_true and all_pred:
            cm_total = confusion_matrix(all_true, all_pred, labels=STAGE_LABELS)
            acc_total = accuracy_score(all_true, all_pred)
            f1_total = f1_score(all_true, all_pred, average="macro")
            print("\n========== Current cumulative result ==========")
            print("Accuracy:", round(acc_total, 3))
            print("Macro-F1:", round(f1_total, 3))
            print("Confusion Matrix:\n", cm_total)
        raise SystemExit("Terminated due to length mismatch.")

    acc = accuracy_score(gt_segment, pred_labels)
    f1 = f1_score(gt_segment, pred_labels, average="macro")
    print(f"  ACC={acc:.3f} | F1={f1:.3f}")

    all_true.extend(gt_segment)
    all_pred.extend(pred_labels)

    record = {
        "subject": psg_file,
        "hypnogram": hyp_file,
        "question": question,
        "response": response,
        "gt_count": n_epochs,
        "pred_count": len(pred_labels),
        "accuracy": acc,
        "f1_macro": f1,
        "messages": agent.messages,
    }

    results.append(record)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(SAVE_DIR, f"{psg_file[:-4]}_{timestamp}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    # print(f"  Saved log to {save_path}")

# ==========================
# summary
# ==========================
if all_true:
    cm_total = confusion_matrix(all_true, all_pred, labels=STAGE_LABELS)
    acc_total = accuracy_score(all_true, all_pred)
    f1_total = f1_score(all_true, all_pred, average="macro")

    summary = {
        "total_subjects": len(results),
        "total_samples": len(all_true),
        "accuracy": acc_total,
        "macro_f1": f1_total,
        "confusion_matrix": cm_total.tolist(),
        "labels": STAGE_LABELS,
    }

    summary_path = os.path.join(SAVE_DIR, "sleep_agent_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== summary ==========")
    print("Accuracy:", round(acc_total, 3))
    print("Macro-F1:", round(f1_total, 3))
    print("Confusion Matrix:")
    print(cm_total)
    print(f"\nSaved summary to {summary_path}")
else:
    print("No valid predictions collected. Check data or preprocessing.")
