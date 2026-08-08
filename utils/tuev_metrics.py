"""Shared event-level TUEV scoring utilities.

Metric policy:
- Positive TUEV annotations are treated as channel-level GT events.
- Raw Agent/tool reports on the same channel are merged into report episodes
  when their temporal gap is small.
- A GT event is counted as a hit when same-channel report episodes cover at
  least the configured fraction of that GT event duration.
- A strict unmatched report is a merged report episode with no positive GT
  overlap on the same channel.
- Strict unmatched reports are split into explicit_negative_reports when they
  overlap an annotated negative event, and unverified_reports when they only
  fall in unannotated TUEV regions.

The strict unmatched report rate is a conservative report-episode-level
statistic. It is useful for auditing extra reports, but it is not the same as a
clinical false alarm frequency measured on a continuous time axis.
"""

import json
import math
from pathlib import Path

COUNT_KEYS = [
    "files",
    "total_gt",
    "total_reports",
    "hits",
    "misses",
    "strict_unmatched_reports",
    "explicit_negative_reports",
    "unverified_reports",
]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_summary(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def aggregate_summaries(summaries):
    totals = {key: 0 for key in COUNT_KEYS}
    thresholds = set()
    models = set()

    for summary in summaries:
        for key in COUNT_KEYS:
            totals[key] += int(summary.get(key, 0))
        if "threshold" in summary:
            thresholds.add(summary["threshold"])
        if "model" in summary:
            models.add(summary["model"])

    total_gt = totals["total_gt"]
    total_reports = totals["total_reports"]
    result = {
        **totals,
        "hit_rate": totals["hits"] / total_gt if total_gt else math.nan,
        "strict_unmatched_report_rate": (
            totals["strict_unmatched_reports"] / total_reports if total_reports else math.nan
        ),
        "explicit_negative_report_rate": (
            totals["explicit_negative_reports"] / total_reports if total_reports else math.nan
        ),
        "unverified_report_rate": (
            totals["unverified_reports"] / total_reports if total_reports else math.nan
        ),
    }
    if thresholds:
        result["thresholds"] = sorted(thresholds)
    if models:
        result["models"] = sorted(models)
    return result


def calculate_overlap(box_a, box_b):
    inter_start = max(box_a[0], box_b[0])
    inter_end = min(box_a[1], box_b[1])
    return max(0, inter_end - inter_start)


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


def merge_predictions(predictions, gap_threshold=1.0):
    grouped = {}
    for pred in predictions:
        grouped.setdefault(pred["channel"], []).append(pred)

    merged = []
    for channel, group in grouped.items():
        current = None
        for pred in sorted(group, key=lambda item: item["start_time"]):
            if current is None:
                current = pred.copy()
            elif pred["start_time"] <= current["end_time"] + gap_threshold:
                current["end_time"] = max(current["end_time"], pred["end_time"])
            else:
                merged.append(current)
                current = pred.copy()
        if current is not None:
            merged.append(current)
    return sorted(merged, key=lambda item: (item["start_time"], item["channel"]))


def score_predictions(gt_events, negative_events, raw_predictions, threshold=0.7, gap_threshold=1.0):
    # Reports are first merged into channel-level episodes. A GT event is hit
    # when reports on the same channel cover enough of that GT duration.
    report_episodes = merge_predictions(raw_predictions, gap_threshold=gap_threshold)
    detected_gt = set()

    for gt_index, gt in enumerate(gt_events):
        gt_length = gt["end"] - gt["start"]
        overlap_intervals = []
        for report in report_episodes:
            if report["channel"] != gt["channel_name"]:
                continue
            overlap = calculate_overlap([gt["start"], gt["end"]], [report["start_time"], report["end_time"]])
            if overlap <= 0:
                continue
            overlap_intervals.append([
                max(gt["start"], report["start_time"]),
                min(gt["end"], report["end_time"]),
            ])
        covered_length = interval_total_length(merge_intervals(overlap_intervals))
        if gt_length > 0 and covered_length / gt_length >= threshold:
            detected_gt.add(gt_index)

    scored_report_episodes = []
    positive_overlapping_reports = 0
    hit_contributing_reports = 0
    strict_unmatched_reports = 0
    explicit_negative_reports = 0
    unverified_reports = 0

    for report in report_episodes:
        report_box = [report["start_time"], report["end_time"]]
        overlapping_gt_indices = [
            gt_index
            for gt_index, gt in enumerate(gt_events)
            if report["channel"] == gt["channel_name"]
            and calculate_overlap([gt["start"], gt["end"]], report_box) > 0
        ]
        has_positive_overlap = bool(overlapping_gt_indices)
        has_negative_overlap = any(
            report["channel"] == neg["channel_name"]
            and calculate_overlap([neg["start"], neg["end"]], report_box) > 0
            for neg in negative_events
        )

        scored_report = report.copy()
        scored_report["overlapping_gt_indices"] = overlapping_gt_indices
        scored_report["is_false_positive"] = not has_positive_overlap
        scored_report["contributes_to_hit"] = any(gt_index in detected_gt for gt_index in overlapping_gt_indices)
        scored_report_episodes.append(scored_report)

        if has_positive_overlap:
            positive_overlapping_reports += 1
            if scored_report["contributes_to_hit"]:
                hit_contributing_reports += 1
            continue

        # Strict unmatched reports are reported episodes without any positive
        # GT overlap on the same channel. They are split into explicit negative
        # overlap and unverified regions because TUEV labels are sparse.
        strict_unmatched_reports += 1
        if has_negative_overlap:
            explicit_negative_reports += 1
        else:
            unverified_reports += 1

    total_gt = len(gt_events)
    total_raw_preds = len(raw_predictions)
    total_preds = len(report_episodes)
    hits = len(detected_gt)

    metrics = {
        "total_gt": total_gt,
        "total_raw_preds": total_raw_preds,
        "total_preds": total_preds,
        "hits": hits,
        "misses": total_gt - hits,
        "positive_overlapping_reports": positive_overlapping_reports,
        "hit_contributing_reports": hit_contributing_reports,
        "no_positive_overlap_predictions": strict_unmatched_reports,
        "false_positives": strict_unmatched_reports,
        "strict_unmatched_reports": strict_unmatched_reports,
        "explicit_negative_reports": explicit_negative_reports,
        "unverified_reports": unverified_reports,
        "hit_rate": hits / total_gt if total_gt else float("nan"),
        "positive_overlap_report_rate": positive_overlapping_reports / total_preds if total_preds else float("nan"),
        "hit_contributing_report_rate": hit_contributing_reports / total_preds if total_preds else float("nan"),
        "false_alarm_rate": strict_unmatched_reports / total_preds if total_preds else float("nan"),
        "strict_unmatched_report_rate": strict_unmatched_reports / total_preds if total_preds else float("nan"),
        "explicit_negative_report_rate": explicit_negative_reports / total_preds if total_preds else float("nan"),
        "unverified_report_rate": unverified_reports / total_preds if total_preds else float("nan"),
    }
    return metrics, report_episodes, scored_report_episodes


def summarize_metrics(metrics, model):
    return {
        "files": 1,
        "model": model,
        "total_gt": metrics["total_gt"],
        "total_reports": metrics["total_preds"],
        "hit_rate": metrics["hit_rate"],
        "strict_unmatched_report_rate": metrics["strict_unmatched_report_rate"],
        "explicit_negative_report_rate": metrics["explicit_negative_report_rate"],
        "unverified_report_rate": metrics["unverified_report_rate"],
        "hits": metrics["hits"],
        "misses": metrics["misses"],
        "strict_unmatched_reports": metrics["strict_unmatched_reports"],
        "explicit_negative_reports": metrics["explicit_negative_reports"],
        "unverified_reports": metrics["unverified_reports"],
    }


def write_file_outputs(out_dir, stem, edf_path, rec_path, model, gt_events, negative_events, raw_predictions,
                       metrics, scored_report_episodes, raw_logs, report_overlap_threshold=0.7):
    file_out_dir = Path(out_dir) / stem
    messages_dir = file_out_dir / "messages"
    file_out_dir.mkdir(parents=True, exist_ok=True)
    messages_dir.mkdir(parents=True, exist_ok=True)
    messages_file = messages_dir / f"{stem}.messages.json"

    with (file_out_dir / "agent_raw.jsonl").open("w", encoding="utf-8") as raw_f:
        for log in raw_logs:
            raw_log = {key: value for key, value in log.items() if key != "messages"}
            raw_log["messages_file"] = str(messages_file)
            raw_f.write(json.dumps(raw_log, ensure_ascii=False, default=str) + "\n")

    write_json(messages_file, {
        "stem": stem,
        "edf": edf_path,
        "rec": rec_path,
        "model": model,
        "candidates": [
            {
                "pair_index": log["pair_index"],
                "stem": log["stem"],
                "edf": log["edf"],
                "rec": log["rec"],
                "candidate_index": log["candidate_index"],
                "window": log["window"],
                "question": log["question"],
                "model": log["model"],
                "messages": log["messages"],
                "parsed_events": log["parsed_events"],
                "error": log["error"],
            }
            for log in raw_logs
        ],
    })

    with (file_out_dir / "agent_predictions.jsonl").open("w", encoding="utf-8") as pred_f:
        pred_f.write(json.dumps({
            "stem": stem,
            "edf": edf_path,
            "rec": rec_path,
            "gt_events": gt_events,
            "negative_events": negative_events,
            "predictions": raw_predictions,
            "scored_report_episodes": scored_report_episodes,
            "metrics": {**metrics, "stem": stem, "edf": edf_path, "rec": rec_path},
            "messages_file": str(messages_file),
        }, ensure_ascii=False, default=str) + "\n")

    write_json(file_out_dir / "metrics.json", {
        "aggregate": {
            "files": 1,
            "model": model,
            "report_overlap_threshold": report_overlap_threshold,
            "gt_policy": "merged_positive_events",
            "prediction_policy": "merged_report_episodes",
            **metrics,
        },
        "per_file": [{**metrics, "stem": stem, "edf": edf_path, "rec": rec_path}],
    })
    write_json(file_out_dir / "summary.json", summarize_metrics(metrics, model))


def write_global_outputs(out_dir):
    root_out_dir = Path(out_dir)
    summary_files = sorted(root_out_dir.glob("*/summary.json"))
    summaries = [load_summary(path) for path in summary_files]
    if summaries:
        aggregate = aggregate_summaries(summaries)
        aggregate["summary_files"] = [str(path) for path in summary_files]
    else:
        aggregate = {
            "files": 0,
            "total_gt": 0,
            "total_reports": 0,
            "hits": 0,
            "misses": 0,
            "strict_unmatched_reports": 0,
            "explicit_negative_reports": 0,
            "unverified_reports": 0,
            "hit_rate": float("nan"),
            "strict_unmatched_report_rate": float("nan"),
            "explicit_negative_report_rate": float("nan"),
            "unverified_report_rate": float("nan"),
            "summary_files": [],
        }
    write_json(root_out_dir / "aggregate_metrics.json", {"aggregate": aggregate, "summaries": summaries})
    write_json(root_out_dir / "aggregate_summary.json", aggregate)
    return aggregate
