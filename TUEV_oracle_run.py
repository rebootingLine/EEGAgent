"""Run tool-only TUEV evaluation with event-level scoring.

This script evaluates the local seizure tool directly, without the LLM Agent.
It scans the annotated TUEV candidate windows, calls the one-second seizure
tool in valid time chunks, converts tool probabilities into channel-time
reports, and scores those reports with the same event-level policy used by the
Agent evaluation.

The oracle result is meant to describe the callable tool's behavior under the
current scoring protocol. It helps separate local tool capacity from LLM-side
tool selection, window handling, and response parsing.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REPORT_OVERLAP_THRESHOLD = 0.7
MERGE_GAP_THRESHOLD = 1.0
POSITIVE_CLASSES = {1, 2, 3}
NEGATIVE_CLASSES = {4, 5, 6}

CHANNEL_MAP = {
    0: "FP1-F7", 1: "F7-T3", 2: "T3-T5", 3: "T5-O1",
    4: "FP2-F8", 5: "F8-T4", 6: "T4-T6", 7: "T6-O2",
    8: "A1-T3", 9: "T3-C3", 10: "C3-CZ", 11: "CZ-C4",
    12: "C4-T4", 13: "T4-A2", 14: "FP1-F3", 15: "F3-C3",
    16: "C3-P3", 17: "P3-O1", 18: "FP2-F4", 19: "F4-C4",
    20: "C4-P4", 21: "P4-O2",
}


def read_rec(rec_path):
    rows = []
    with open(rec_path, newline="") as f:
        for channel, start, end, label in csv.reader(f):
            channel = int(channel)
            rows.append({
                "channel": channel,
                "channel_name": CHANNEL_MAP.get(channel, str(channel)),
                "start": float(start),
                "end": float(end),
                "class": int(label),
            })
    return rows


def merge_rows(rows, gap_threshold=MERGE_GAP_THRESHOLD):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["channel"], row["class"])].append(row)

    merged = []
    for (_channel, _label), group in grouped.items():
        current = None
        for row in sorted(group, key=lambda item: item["start"]):
            if current is None:
                current = row.copy()
            elif row["start"] <= current["end"] + gap_threshold:
                current["end"] = max(current["end"], row["end"])
            else:
                merged.append(current)
                current = row.copy()
        if current is not None:
            merged.append(current)
    return sorted(merged, key=lambda item: (item["start"], item["channel"], item["class"]))


def build_candidate_windows(merged_rows, gap_threshold=MERGE_GAP_THRESHOLD):
    windows = []
    current = None
    for row in sorted(merged_rows, key=lambda item: item["start"]):
        if current is None:
            current = {"start": row["start"], "end": row["end"]}
        elif row["start"] <= current["end"] + gap_threshold:
            current["end"] = max(current["end"], row["end"])
        else:
            windows.append(current)
            current = {"start": row["start"], "end": row["end"]}
    if current is not None:
        windows.append(current)
    return windows


def find_rec_edf_pairs(data_dir, file_list=None):
    data_dir = Path(data_dir)
    rec_paths = sorted(data_dir.rglob("*.rec"))
    edf_by_stem = {path.stem: path for path in data_dir.rglob("*.edf")}

    allowed = None
    if file_list:
        with open(file_list, encoding="utf-8") as f:
            allowed = {line.strip() for line in f if line.strip() and not line.startswith("#")}

    pairs = []
    for rec_path in rec_paths:
        edf_path = edf_by_stem.get(rec_path.stem)
        if not edf_path:
            continue
        if allowed and rec_path.stem not in allowed and rec_path.name not in allowed and edf_path.name not in allowed:
            continue
        pairs.append({"stem": rec_path.stem, "rec": str(rec_path), "edf": str(edf_path)})
    return pairs


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def overlap_seconds(gt, pred):
    return max(0.0, min(gt["end"], pred["end_time"]) - max(gt["start"], pred["start_time"]))


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def interval_total_length(intervals):
    return sum(end - start for start, end in intervals)


def merge_predictions(predictions, gap_threshold=MERGE_GAP_THRESHOLD):
    grouped = defaultdict(list)
    for pred in predictions:
        grouped[pred["channel"]].append(pred)

    merged = []
    for channel, group in grouped.items():
        current = None
        for pred in sorted(group, key=lambda item: item["start_time"]):
            if current is None:
                current = pred.copy()
                current["raw_report_count"] = 1
                current["raw_reports"] = [pred.copy()]
                continue
            if pred["start_time"] <= current["end_time"] + gap_threshold:
                current["end_time"] = max(current["end_time"], pred["end_time"])
                current["max_prob"] = max(current.get("max_prob", 0.0), pred.get("max_prob", 0.0))
                current["raw_report_count"] += 1
                current["raw_reports"].append(pred.copy())
            else:
                merged.append(current)
                current = pred.copy()
                current["raw_report_count"] = 1
                current["raw_reports"] = [pred.copy()]
        if current is not None:
            merged.append(current)

    return sorted(merged, key=lambda item: (item["start_time"], item["channel"]))


def score_reports(gt_events, predictions, threshold=REPORT_OVERLAP_THRESHOLD):
    report_episodes = merge_predictions(predictions)
    detected_gt = set()
    scored_predictions = []
    no_positive_overlap_predictions = 0

    for gt_index, gt in enumerate(gt_events):
        gt_length = gt["end"] - gt["start"]
        overlap_intervals = []
        overlapping_report_indices = []
        for report_index, report in enumerate(report_episodes):
            if report["channel"] != gt["channel_name"]:
                continue
            overlap = overlap_seconds(gt, report)
            if overlap <= 0:
                continue
            overlap_intervals.append((
                max(gt["start"], report["start_time"]),
                min(gt["end"], report["end_time"]),
            ))
            overlapping_report_indices.append(report_index)

        covered_length = interval_total_length(merge_intervals(overlap_intervals))
        coverage = covered_length / gt_length if gt_length > 0 else 0.0
        if coverage >= threshold:
            detected_gt.add(gt_index)

    positive_overlapping_reports = 0
    hit_contributing_reports = 0
    for report_index, report in enumerate(report_episodes):
        overlapping_gt_indices = []
        best_overlap = 0.0
        for gt_index, gt in enumerate(gt_events):
            if report["channel"] != gt["channel_name"]:
                continue
            overlap = overlap_seconds(gt, report)
            if overlap > 0:
                overlapping_gt_indices.append(gt_index)
                best_overlap = max(best_overlap, overlap)

        scored_report = report.copy()
        scored_report["overlapping_gt_indices"] = overlapping_gt_indices
        scored_report["best_positive_overlap_seconds"] = best_overlap
        scored_report["is_false_positive"] = not overlapping_gt_indices
        scored_report["contributes_to_hit"] = any(gt_index in detected_gt for gt_index in overlapping_gt_indices)
        scored_predictions.append(scored_report)

        if scored_report["is_false_positive"]:
            no_positive_overlap_predictions += 1
        else:
            positive_overlapping_reports += 1
            if scored_report["contributes_to_hit"]:
                hit_contributing_reports += 1

    total_gt = len(gt_events)
    total_raw_preds = len(predictions)
    total_preds = len(report_episodes)
    hits = len(detected_gt)
    false_positives = no_positive_overlap_predictions
    return {
        "total_gt": total_gt,
        "total_raw_preds": total_raw_preds,
        "total_preds": total_preds,
        "hits": hits,
        "misses": total_gt - hits,
        "positive_overlapping_reports": positive_overlapping_reports,
        "hit_contributing_reports": hit_contributing_reports,
        "no_positive_overlap_predictions": no_positive_overlap_predictions,
        "false_positives": false_positives,
        "hit_rate": hits / total_gt if total_gt else math.nan,
        "positive_overlap_report_rate": positive_overlapping_reports / total_preds if total_preds else math.nan,
        "hit_contributing_report_rate": hit_contributing_reports / total_preds if total_preds else math.nan,
        "false_alarm_rate": false_positives / total_preds if total_preds else math.nan,
        "no_positive_overlap_rate": no_positive_overlap_predictions / total_preds if total_preds else math.nan,
        "scored_report_episodes": scored_predictions,
    }


def count_negative_overlaps(negative_events, report_episodes):
    explicit_negative_reports = 0
    unverified_reports = 0

    for report in report_episodes:
        has_negative_overlap = False
        for neg in negative_events:
            if report["channel"] != neg["channel_name"]:
                continue
            if overlap_seconds(neg, report) > 0:
                has_negative_overlap = True
                break
        if has_negative_overlap:
            explicit_negative_reports += 1
        else:
            unverified_reports += 1

    return explicit_negative_reports, unverified_reports


def run_oracle(pairs, config_path, threshold, out_dir):
    from tools.dataLoad import dataLoad
    from tools.registerData import registerData
    from tools.singleChannel import seizureNormalModel_OneSecond

    config = load_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "oracle_raw.jsonl"
    pred_path = out_dir / "oracle_predictions.jsonl"

    totals = defaultdict(int)
    per_file = []

    with raw_path.open("w", encoding="utf-8") as raw_f, pred_path.open("w", encoding="utf-8") as pred_f:
        for pair in pairs:
            rows = read_rec(pair["rec"])
            merged_rows = merge_rows(rows)
            candidate_windows = build_candidate_windows(merged_rows)
            gt_events = [row for row in merged_rows if row["class"] in POSITIVE_CLASSES]
            negative_events = [row for row in merged_rows if row["class"] in NEGATIVE_CLASSES]

            data = dataLoad(pair["edf"], config)
            registerData(data)

            predictions = []
            for window_index, window in enumerate(candidate_windows):
                start = int(math.floor(window["start"]))
                end = int(math.ceil(window["end"]))
                if end <= start:
                    end = start + 1

                for chunk_start in range(start, end, 10):
                    chunk_end = min(chunk_start + 10, end)
                    result = seizureNormalModel_OneSecond(
                        name=list(CHANNEL_MAP.values()),
                        start=chunk_start,
                        end=chunk_end,
                        config=config,
                    )
                    raw_f.write(json.dumps({
                        "stem": pair["stem"],
                        "edf": pair["edf"],
                        "rec": pair["rec"],
                        "candidate_index": window_index,
                        "window": window,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "result": result,
                    }, ensure_ascii=False) + "\n")

                    for item in result:
                        duration = item["duration"].replace("s", "")
                        sec_start, sec_end = [int(value) for value in duration.split("-")]
                        for channel in CHANNEL_MAP.values():
                            prob = item[channel]["seiz"]
                            if prob >= threshold:
                                predictions.append({
                                    "channel": channel,
                                    "start_time": float(sec_start),
                                    "end_time": float(sec_end),
                                    "max_prob": prob,
                                })

            metrics = score_reports(gt_events, predictions)
            unmatched_reports = [
                report for report in metrics["scored_report_episodes"] if report["is_false_positive"]
            ]
            explicit_negative_reports, unverified_reports = count_negative_overlaps(
                negative_events, unmatched_reports
            )
            metrics["strict_unmatched_reports"] = metrics["false_positives"]
            metrics["strict_unmatched_report_rate"] = metrics["false_alarm_rate"]
            metrics["explicit_negative_reports"] = explicit_negative_reports
            metrics["unverified_reports"] = unverified_reports
            metrics["explicit_negative_report_rate"] = (
                explicit_negative_reports / metrics["total_preds"] if metrics["total_preds"] else math.nan
            )
            metrics["unverified_report_rate"] = (
                unverified_reports / metrics["total_preds"] if metrics["total_preds"] else math.nan
            )
            summary = {key: value for key, value in metrics.items() if key != "scored_report_episodes"}
            summary.update({"stem": pair["stem"], "edf": pair["edf"], "rec": pair["rec"]})
            per_file.append(summary)
            for key in [
                "total_gt",
                "total_raw_preds",
                "total_preds",
                "hits",
                "misses",
                "positive_overlapping_reports",
                "hit_contributing_reports",
                "no_positive_overlap_predictions",
                "false_positives",
                "strict_unmatched_reports",
                "explicit_negative_reports",
                "unverified_reports",
            ]:
                totals[key] += int(summary[key])

            pred_f.write(json.dumps({
                "stem": pair["stem"],
                "edf": pair["edf"],
                "rec": pair["rec"],
                "gt_events": gt_events,
                "negative_events": negative_events,
                "raw_predictions": predictions,
                "scored_report_episodes": metrics["scored_report_episodes"],
                "metrics": summary,
            }, ensure_ascii=False) + "\n")

    aggregate = {
        "files": len(pairs),
        "threshold": threshold,
        "report_overlap_threshold": REPORT_OVERLAP_THRESHOLD,
        "gt_policy": "merged_positive_events",
        "prediction_policy": "merged_report_episodes",
        "total_gt": totals["total_gt"],
        "total_raw_preds": totals["total_raw_preds"],
        "total_preds": totals["total_preds"],
        "hits": totals["hits"],
        "misses": totals["misses"],
        "positive_overlapping_reports": totals["positive_overlapping_reports"],
        "hit_contributing_reports": totals["hit_contributing_reports"],
        "no_positive_overlap_predictions": totals["no_positive_overlap_predictions"],
        "false_positives": totals["false_positives"],
        "strict_unmatched_reports": totals["strict_unmatched_reports"],
        "explicit_negative_reports": totals["explicit_negative_reports"],
        "unverified_reports": totals["unverified_reports"],
        "hit_rate": totals["hits"] / totals["total_gt"] if totals["total_gt"] else math.nan,
        "positive_overlap_report_rate": (
            totals["positive_overlapping_reports"] / totals["total_preds"] if totals["total_preds"] else math.nan
        ),
        "hit_contributing_report_rate": (
            totals["hit_contributing_reports"] / totals["total_preds"] if totals["total_preds"] else math.nan
        ),
        "false_alarm_rate": totals["false_positives"] / totals["total_preds"] if totals["total_preds"] else math.nan,
        "strict_unmatched_report_rate": (
            totals["strict_unmatched_reports"] / totals["total_preds"] if totals["total_preds"] else math.nan
        ),
        "explicit_negative_report_rate": (
            totals["explicit_negative_reports"] / totals["total_preds"] if totals["total_preds"] else math.nan
        ),
        "unverified_report_rate": (
            totals["unverified_reports"] / totals["total_preds"] if totals["total_preds"] else math.nan
        ),
        "no_positive_overlap_rate": (
            totals["no_positive_overlap_predictions"] / totals["total_preds"] if totals["total_preds"] else math.nan
        ),
    }
    summary = {
        "files": aggregate["files"],
        "threshold": aggregate["threshold"],
        "total_gt": aggregate["total_gt"],
        "total_reports": aggregate["total_preds"],
        "hit_rate": aggregate["hit_rate"],
        "strict_unmatched_report_rate": aggregate["strict_unmatched_report_rate"],
        "explicit_negative_report_rate": aggregate["explicit_negative_report_rate"],
        "unverified_report_rate": aggregate["unverified_report_rate"],
        "hits": aggregate["hits"],
        "misses": aggregate["misses"],
        "strict_unmatched_reports": aggregate["strict_unmatched_reports"],
        "explicit_negative_reports": aggregate["explicit_negative_reports"],
        "unverified_reports": aggregate["unverified_reports"],
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "per_file": per_file}, f, ensure_ascii=False, indent=2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Clean TUEV oracle evaluation with merged GT and unmerged reports.")
    parser.add_argument("--mode", choices=["list", "oracle"], default="oracle")
    parser.add_argument("--data-dir", required=True, help="Directory containing paired .edf/.rec files.")
    parser.add_argument("--file-list", default=None, help="Optional stem/.edf/.rec list.")
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--threshold", type=float, default=0.7, help="Seizure probability threshold.")
    parser.add_argument("--out-dir", default="runs/tuev_oracle_run")
    args = parser.parse_args()

    pairs = find_rec_edf_pairs(args.data_dir, args.file_list)
    print(f"Found {len(pairs)} paired .rec/.edf files.")
    for pair in pairs:
        print(f"{pair['stem']}: rec={pair['rec']} edf={pair['edf']}")

    if args.mode == "list":
        return

    aggregate = run_oracle(pairs, Path(args.config), args.threshold, Path(args.out_dir))
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))

'''
python3 TUEV_oracle_run.py --data-dir xxx/TUEV/edf/eval 指定TUEV数据集的测试集或者其他子集
--out-dir 指定结果存储路径
'''
if __name__ == "__main__":
    main()
