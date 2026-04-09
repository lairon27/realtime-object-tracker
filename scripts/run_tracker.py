"""
CLI runner for the real-time object tracking pipeline.

Usage examples:
    # Track a video file, save result
    python scripts/run_tracker.py --input video.mp4 --output result.mp4

    # Track a MOT17 sequence, display live
    python scripts/run_tracker.py --mot17 data/MOT17\ 2/train/MOT17-02-DPM --show

    # Use ByteTrack with custom config
    python scripts/run_tracker.py --input video.mp4 --tracker bytetrack --config configs/default.yaml

    # Just display, no save
    python scripts/run_tracker.py --input video.mp4 --show --no-save
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.tracker import TrackingPipeline
from src.visualization.draw import resize_for_display


def make_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time multi-object tracking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input",  type=str, help="Path to input video file")
    source.add_argument("--mot17",  type=str, help="Path to MOT17 sequence directory")

    parser.add_argument("--output",  type=str, default=None,
                        help="Output video path (default: <input>_tracked.mp4)")
    parser.add_argument("--tracker", type=str, default="bytetrack",
                        choices=["sort", "bytetrack"],
                        help="Tracker algorithm")
    parser.add_argument("--config",  type=str, default="configs/default.yaml",
                        help="Config YAML path")
    parser.add_argument("--show",    action="store_true",
                        help="Display frames in a window (requires display)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save output video")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames")
    parser.add_argument("--show-dets",  action="store_true",
                        help="Also draw raw detections (grey boxes)")
    args = parser.parse_args()

    # Build pipeline
    pipeline = TrackingPipeline.from_config(args.config, tracker_override=args.tracker)
    pipeline.show_dets = args.show_dets

    # Determine output path
    save_video = not args.no_save
    output_path = args.output
    if save_video and output_path is None:
        if args.input:
            stem = Path(args.input).stem
            output_path = str(Path(args.input).parent / f"{stem}_tracked.mp4")
        else:
            seq_name = Path(args.mot17).name
            output_path = f"outputs/{seq_name}_tracked.mp4"
            os.makedirs("outputs", exist_ok=True)

    writer = None
    total_frames = 0
    t_start = time.time()

    print(f"Tracker  : {args.tracker}")
    print(f"Config   : {args.config}")
    if args.input:
        print(f"Input    : {args.input}")
        frame_gen = pipeline.run_video(args.input, max_frames=args.max_frames)
        # Peek at first frame to get size
        cap_tmp = cv2.VideoCapture(args.input)
        fps_src = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_tmp.release()
    else:
        print(f"MOT17 seq: {args.mot17}")
        frame_gen = pipeline.run_mot17(args.mot17, max_frames=args.max_frames)
        from src.data_loader import MOT17Sequence
        info = MOT17Sequence(args.mot17).info
        fps_src = float(info.frame_rate)
        w, h = info.im_width, info.im_height

    print(f"Output   : {output_path if save_video else '(not saving)'}")
    print(f"Source   : {w}x{h} @ {fps_src:.1f} fps\n")

    for frame_id, annotated, tracks in frame_gen:
        total_frames += 1

        if save_video:
            if writer is None:
                writer = make_video_writer(output_path, fps_src, annotated.shape[1], annotated.shape[0])
            writer.write(annotated)

        if args.show:
            disp = resize_for_display(annotated, max_width=1280)
            cv2.imshow("Tracking", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

        if total_frames % 50 == 0:
            elapsed = time.time() - t_start
            fps_proc = total_frames / elapsed
            n_tracks = len(tracks)
            print(f"  Frame {frame_id:4d}  |  tracks={n_tracks:2d}  |  {fps_proc:.1f} fps")

    # Cleanup
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\nDone. {total_frames} frames in {elapsed:.1f}s "
          f"({total_frames/elapsed:.1f} fps avg)")
    if save_video:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
