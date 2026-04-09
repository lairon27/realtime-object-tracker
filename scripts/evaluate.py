"""
Evaluate SORT / ByteTrack / DeepSORT on MOT17 train sequences.

Usage:
    python scripts/evaluate.py [--detector DPM|FRCNN|SDP] [--sequences SEQ [SEQ ...]]
                               [--trackers sort bytetrack deepsort]
                               [--use-gt-dets]  # use pre-computed det.txt instead of YOLOv8

Examples:
    # All DPM sequences, all trackers, YOLOv8 detections
    python scripts/evaluate.py --detector DPM

    # Use pre-computed GT detections (fast, isolates tracker quality)
    python scripts/evaluate.py --detector DPM --use-gt-dets

    # Single sequence, specific tracker
    python scripts/evaluate.py --sequences MOT17-02-DPM --trackers bytetrack
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import MOT17Dataset, MOT17Sequence, GroundTruth, Detection
from src.tracking.sort import SORTTracker
from src.tracking.bytetrack import ByteTracker
from src.evaluation.metrics import MOTAccumulator, format_summary, merge_results

 
# Helpers

def gt_to_tracked(gt: GroundTruth):
    """Wrap a GroundTruth as a minimal tracked object for the accumulator."""
    class _Obj:
        def __init__(self, tid, tlbr):
            self.track_id = tid
            self.tlbr = tlbr
    return _Obj(gt.track_id, gt.tlbr)


def det_to_array(dets: list) -> np.ndarray:
    """Convert list of Detection objects to (N, 5) array [x1,y1,x2,y2,conf]."""
    if not dets:
        return np.empty((0, 5), dtype=np.float32)
    rows = []
    for d in dets:
        tlbr = d.tlbr
        rows.append([tlbr[0], tlbr[1], tlbr[2], tlbr[3], d.confidence])
    return np.array(rows, dtype=np.float32)


def build_tracker(name: str):
    if name == "sort":
        return SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    if name == "bytetrack":
        return ByteTracker(high_thresh=0.5, low_thresh=0.1,
                           match_thresh=0.8, max_age=30, min_hits=3)
    raise ValueError(f"Unknown tracker: {name}")


def run_sequence(
    seq: MOT17Sequence,
    tracker_name: str,
    use_gt_dets: bool,
    yolo_detector=None,
) -> dict:
    """Run one tracker on one sequence and return metrics dict."""
    tracker = build_tracker(tracker_name)
    acc = MOTAccumulator(iou_threshold=0.5)

    for fid, img, mot_dets, gts in seq.iter_frames():
        # detections  
        if use_gt_dets:
            inp = det_to_array(mot_dets)          # pre-computed det.txt
        else:
            raw = yolo_detector.detect(img)        # YOLOv8
            inp = raw[:, :5] if len(raw) else np.empty((0, 5), dtype=np.float32)

        # update tracker  
        confirmed = tracker.update(inp)

        # build GT wrappers (active pedestrians only)  
        gt_objs = [gt_to_tracked(g) for g in gts
                   if g.is_active == 1 and g.cls == 1]  # class 1 = pedestrian in MOT17

        acc.update(frame_id=fid, gt_objects=gt_objs, tracked_objects=confirmed)

    return acc.compute()


# Main

def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT trackers on MOT17")
    parser.add_argument("--data-root",   default="data/MOT17 2",
                        help="Path to MOT17 dataset root")
    parser.add_argument("--split",       default="train")
    parser.add_argument("--detector",    default="DPM",
                        choices=["DPM", "FRCNN", "SDP"],
                        help="MOT17 detector suffix to filter sequences")
    parser.add_argument("--sequences",   nargs="+", default=None,
                        help="Specific sequence names; default = all matching --detector")
    parser.add_argument("--trackers",    nargs="+",
                        default=["sort", "bytetrack"],
                        choices=["sort", "bytetrack"])
    parser.add_argument("--use-gt-dets", action="store_true",
                        help="Use pre-computed det.txt detections (fast)")
    parser.add_argument("--yolo-model",  default="yolov8n.pt",
                        help="YOLOv8 model weights (ignored with --use-gt-dets)")
    args = parser.parse_args()

    # Load dataset
    dataset = MOT17Dataset(args.data_root, split=args.split)

    # Filter sequences
    if args.sequences:
        seqs = [dataset.get_sequence(n) for n in args.sequences]
    else:
        seqs = [s for s in dataset.sequences if s.name.endswith(args.detector)]

    if not seqs:
        print(f"No sequences found for detector={args.detector}")
        sys.exit(1)

    print(f"\nDataset : {args.data_root}  split={args.split}")
    print(f"Sequences ({len(seqs)}): {[s.name for s in seqs]}")
    print(f"Trackers : {args.trackers}")
    det_src = "det.txt (pre-computed)" if args.use_gt_dets else f"YOLOv8 ({args.yolo_model})"
    print(f"Detection source: {det_src}\n")

    # Optionally load YOLOv8
    yolo = None
    if not args.use_gt_dets:
        from src.detection.detector import YOLOv8Detector
        yolo = YOLOv8Detector(model_path=args.yolo_model, confidence=0.25, device="cpu")

    # Run evaluation
    all_results: dict[str, list] = {t: [] for t in args.trackers}

    header = f"{'Sequence':<22} {'Tracker':<12} {'MOTA':>7} {'IDF1':>7} {'IDS':>6} {'FP':>6} {'FN':>7} {'MT':>4} {'ML':>4} {'Rec':>6} {'Prec':>6}"
    print(header)
    print("-" * len(header))

    for seq in seqs:
        for tracker_name in args.trackers:
            t0 = time.time()
            res = run_sequence(seq, tracker_name, args.use_gt_dets, yolo)
            elapsed = time.time() - t0
            all_results[tracker_name].append(res)

            print(
                f"{seq.name:<22} {tracker_name:<12} "
                f"{res['mota']:>+7.3f} {res['idf1']:>7.3f} "
                f"{res['ids']:>6d} {res['fp']:>6d} {res['fn']:>7d} "
                f"{res['mt']:>4d} {res['ml']:>4d} "
                f"{res['recall']:>6.3f} {res['precision']:>6.3f}  "
                f"({elapsed:.1f}s)"
            )

    # Dataset-level summary
    if len(seqs) > 1:
        print("\n" + "=" * len(header))
        print(f"{'OVERALL':^{len(header)}}")
        print("=" * len(header))
        for tracker_name in args.trackers:
            merged = merge_results(all_results[tracker_name])
            print(
                f"{'ALL seqs':<22} {tracker_name:<12} "
                f"{merged['mota']:>+7.3f} {merged['idf1']:>7.3f} "
                f"{merged['ids']:>6d} {merged['fp']:>6d} {merged['fn']:>7d} "
                f"{merged['mt']:>4d} {merged['ml']:>4d} "
                f"{merged['recall']:>6.3f} {merged['precision']:>6.3f}"
            )


if __name__ == "__main__":
    main()
