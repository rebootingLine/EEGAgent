from collections import defaultdict
import pandas as pd
import glob
import os

# ================= CONFIG =================
IOU_THRESHOLD = 0.7
MERGE_GAP_THRESHOLD = 1.0
POSITIVE_CLASSES = [1, 2, 3]

CHANNEL_MAP = {
    0: 'FP1-F7', 1: 'F7-T3', 2: 'T3-T5', 3: 'T5-O1',
    4: 'FP2-F8', 5: 'F8-T4', 6: 'T4-T6', 7: 'T6-O2',
    8: 'A1-T3', 9: 'T3-C3', 10: 'C3-CZ', 11: 'CZ-C4',
    12: 'C4-T4', 13: 'T4-A2', 14: 'FP1-F3', 15: 'F3-C3',
    16: 'C3-P3', 17: 'P3-O1', 18: 'FP2-F4', 19: 'F4-C4',
    20: 'C4-P4', 21: 'P4-O2'
}
# ==========================================

def build_questions(folder):
    questions = []
    pairs = find_rec_edf_pairs(folder)

    for rec_path, edf_path in pairs:
        merged_df, candidates = process_rec_annotations(rec_path)

        for idx, (s, e) in enumerate(candidates):
            question = (
                f"If this EEG contains epileptic seizures in [{s:.2f}, {e:.2f}], "
                f"tell me the time and channel. "
                f"Return the results in the format: (channel name, time start, time end)."
            )
            questions.append({
                "edf": edf_path,
                "candidate_index": idx,
                "x": s,
                "y": e,
                "question": question
            })

    return questions

def find_rec_edf_pairs(folder):
    rec_files = glob.glob(os.path.join(folder, "**", "*.rec"), recursive=True)
    edf_files = glob.glob(os.path.join(folder, "**", "*.edf"), recursive=True)
    edf_dict = {os.path.basename(edf): edf for edf in edf_files}

    pairs = []
    for rec in rec_files:
        rec_name = os.path.basename(rec).replace(".rec", ".edf")
        if rec_name in edf_dict:
            pairs.append((rec, edf_dict[rec_name]))

    return pairs

def process_rec_annotations(input_file, output_dir='candidates', gap_threshold=1.0):
    os.makedirs(output_dir, exist_ok=True)

    ### Read original labels ###
    df = pd.read_csv(input_file, header=None, names=['channel', 'start', 'end', 'class'])
    df['channel'] = df['channel'].astype(int)
    df['class'] = df['class'].astype(int)

    ### Merge labels (by channel + class) ###
    merged = []
    for (ch, cls), sub in df.groupby(['channel', 'class']):
        sub = sub.sort_values('start')
        current_start, current_end = None, None
        for _, row in sub.iterrows():
            s, e = row['start'], row['end']
            if current_start is None:
                current_start, current_end = s, e
            elif s <= current_end + gap_threshold:
                current_end = max(current_end, e)
            else:
                merged.append((ch, current_start, current_end, cls))
                current_start, current_end = s, e
        if current_start is not None:
            merged.append((ch, current_start, current_end, cls))

    merged_df = pd.DataFrame(merged, columns=['channel', 'start', 'end', 'class'])

    ### Divide candidate intervals based on the merged labels ###
    candidates = []
    curr_start, curr_end = None, None
    for s, e in merged_df[['start', 'end']].sort_values('start').values.tolist():
        if curr_start is None:
            curr_start, curr_end = s, e
        elif s <= curr_end + gap_threshold:
            curr_end = max(curr_end, e)
        else:
            candidates.append((curr_start, curr_end))
            curr_start, curr_end = s, e
    if curr_start is not None:
        candidates.append((curr_start, curr_end))

    ### Save each candidate interval, containing only the 'merged label' ###
    for i, (s_win, e_win) in enumerate(candidates):
        subset = merged_df[(merged_df['end'] > s_win) & (merged_df['start'] < e_win)].sort_values(['channel', 'start'])
        file_path = os.path.join(output_dir, f'candidate_{i}.rec')
        with open(file_path, 'w') as f:
            f.write(f"{s_win:.2f},{e_win:.2f}\n")
            for _, row in subset.iterrows():
                # class 映射：1,2,3 -> 1；4,5,6 -> 0
                mapped_class = 1 if row['class'] in [1, 2, 3] else 0
                f.write(f"{row['channel']},{row['start']:.2f},{row['end']:.2f},{mapped_class}\n")

    return merged_df, candidates

def calculate_iou(boxA, boxB):
    inter_start = max(boxA[0], boxB[0])
    inter_end = min(boxA[1], boxB[1])
    intersection = max(0, inter_end - inter_start)
    union = (boxA[1]-boxA[0]) + (boxB[1]-boxB[0]) - intersection
    return intersection / union if union > 0 else 0

def merge_predictions(predictions, gap_threshold=MERGE_GAP_THRESHOLD):
    if not predictions: return []
    df = pd.DataFrame(predictions)
    df.rename(columns={'start_time': 'start', 'end_time': 'end'}, inplace=True)
    merged = []
    for channel, sub in df.groupby('channel'):
        sub = sub.sort_values('start')
        current_start, current_end = None, None
        for _, row in sub.iterrows():
            s, e = row['start'], row['end']
            if current_start is None:
                current_start, current_end = s, e
            elif s <= current_end + gap_threshold:
                current_end = max(current_end, e)
            else:
                merged.append({'channel': channel, 'start_time': current_start, 'end_time': current_end})
                current_start, current_end = s, e
        if current_start is not None:
            merged.append({'channel': channel, 'start_time': current_start, 'end_time': current_end})
    return merged

def process_and_merge_rec_file(input_file, gap_threshold=MERGE_GAP_THRESHOLD):
    df = pd.read_csv(input_file, header=None, names=['channel', 'start', 'end', 'class'])
    df['channel'] = df['channel'].astype(int)
    df['class'] = df['class'].astype(int)
    merged = []
    for (ch, cls), sub in df.groupby(['channel', 'class']):
        sub = sub.sort_values('start')
        current_start, current_end = None, None
        for _, row in sub.iterrows():
            s, e = row['start'], row['end']
            if current_start is None:
                current_start, current_end = s, e
            elif s <= current_end + gap_threshold:
                current_end = max(current_end, e)
            else:
                merged.append((ch, current_start, current_end, cls))
                current_start, current_end = s, e
        if current_start is not None:
            merged.append((ch, current_start, current_end, cls))
    return pd.DataFrame(merged, columns=['channel', 'start', 'end', 'class'])

def load_ground_truth(data_folder):
    gt_data = defaultdict(list)
    rec_files = glob.glob(os.path.join(data_folder, "**", "*.rec"), recursive=True)
    for rec_path in rec_files:
        edf_path = rec_path.replace(".rec", ".edf")
        merged_df = process_and_merge_rec_file(rec_path)
        if merged_df.empty: continue
        positive = merged_df[merged_df['class'].isin(POSITIVE_CLASSES)].copy()
        positive['channel_name'] = positive['channel'].map(CHANNEL_MAP)
        positive.dropna(subset=['channel_name'], inplace=True)
        gt_data[edf_path] = positive.to_dict('records')
    return gt_data

# ==================== MAIN LOOP ====================
from tqdm import tqdm
results_storage = defaultdict(list)  
questions = build_questions("./data/edf")
ground_truth = load_ground_truth('./data/edf')


gap = len(questions)
for q in tqdm(questions[:gap]):
    import time
    time.sleep(5)  
    try:
        agent = EEGAgent(config_path="config/config.json", 
                         file_name=q['edf'], 
                         api_key = "***", 
                         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        user_question = f'''Please find all epileptic seizures in this EEG between {round(q['x'])} seconds and {round(q['y'])} seconds. 
Check all channels. For each detected seizure, return exactly one line in this format:
(channel_name, time_start, time_end)
Do not include any extra text, explanation, or commentary. 
Each line should correspond to one seizure event. List all events for all channels'''
        result = agent.run(user_question)
        raw_response = result['response']
        extracted_events = pattern.findall(raw_response)
        extracted_events_processed = [{'channel': e[0], 'start_time': float(e[1]), 'end_time': float(e[2])} for e in extracted_events]
        results_storage[q['edf']].append(extracted_events_processed)
    except Exception as e:
        results_storage[q['edf']].append([])

# ==================== ANALYSIS ====================
total_preds, total_gt, hits = 0, 0, 0
for edf_path, predictions_list in results_storage.items():
    gt_labels = ground_truth.get(edf_path, [])
    total_gt += len(gt_labels)
    merged_preds = merge_predictions([p for sublist in predictions_list for p in sublist])
    total_preds += len(merged_preds)
    
    matched = [False]*len(merged_preds)
    for gt in gt_labels:
        gt_box = [gt['start'], gt['end']]
        for i, pred in enumerate(merged_preds):
            if matched[i]: continue
            if pred['channel'] == gt['channel_name'] and calculate_iou(gt_box, [pred['start_time'], pred['end_time']]) >= IOU_THRESHOLD:
                hits += 1
                matched[i] = True
                break

print("Analysis complete.")
print(f"Total Predictions: {total_preds}")
print(f"Total Ground Truths: {total_gt}")
print(f"Hits (TP): {hits}")
print(f"Misses (FN): {total_gt - hits}")
print(f"False Positives (FP): {total_preds - hits}")
