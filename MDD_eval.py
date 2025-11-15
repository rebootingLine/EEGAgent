import os
import json
from datetime import datetime
from main import EEGAgent  
import time

test_subjects = [
    "H_S5","MDD_S10","MDD_S11","H_S21","MDD_S28",
    "MDD_S14","H_S20","H_S24","MDD_S6","MDD_S30",
    "H_S14","H_S25","MDD_S5","MDD_S21","H_S26",
    "MDD_S33","H_S23","H_S15"
]

data_dir = "./eval/MDD/raw"  
save_dir = "./eval/eval_logs"   
os.makedirs(save_dir, exist_ok=True)

# === file list ===
subject_files = []
for subj in test_subjects:
    group, sid = subj.split("_")
    fname = f"{group} {sid} EC.edf"
    label = 0 if group == "H" else 1
    subject_files.append({
        "name": subj,
        "file": fname,
        "label": label
    })

# === evaluation ===
results = []
question = "Determine whether the subject is a patient with MDD. Please respond with Yes or No."

for subj in subject_files:
    time.sleep(5)
    print(f"\nEvaluating {subj['name']} ...")

    ### init EEGAgent ###
    agent = EEGAgent(
        config_path="config/config.json",
        file_name=subj["file"],
        api_key="***",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    response = agent.run(question)
    print(f"Model response: {response}")

    response_clean = response.strip().lower()
    if "yes" in response_clean:
        pred = 1
    elif "no" in response_clean:
        pred = 0
    else:
        pred = None 

    record = {
        "name": subj["name"],
        "file": subj["file"],
        "label": subj["label"],
        "pred": pred,
        "response": response,
        "question": question,
        "messages": agent.messages
    }
    results.append(record)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "Yes" if pred == 1 else ("No" if pred == 0 else "Unknown")
    save_path = os.path.join(save_dir, f"{subj['name']}_{tag}_{timestamp}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"Saved full conversation to {save_path}")

### === summary === ###
summary_path = os.path.join(save_dir, "all_results_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n📊 All results summary saved to {summary_path}")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

y_true = [r["label"] for r in results if r["pred"] is not None]
y_pred = [r["pred"] for r in results if r["pred"] is not None]

if len(y_true) > 0:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.3f}")
    print(f"F1-score: {f1:.3f}")
else:
    print("\nNo valid predictions to evaluate.")
