"""
MOT17 dataset loader.

GT format  (gt/gt.txt):  frame, id, x, y, w, h, is_active, class, visibility
Det format (det/det.txt): frame, id, x, y, w, h, confidence, -1, -1, -1

All bounding boxes are in (x, y, w, h) format as stored in the files.
Helper methods return them as (x1, y1, x2, y2) = tlbr when requested.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    """Single detection entry from det/det.txt."""
    frame: int
    x: float
    y: float
    w: float
    h: float
    confidence: float

    @property
    def tlbr(self) -> np.ndarray:
        """Return [x1, y1, x2, y2]."""
        return np.array([self.x, self.y, self.x + self.w, self.y + self.h], dtype=np.float32)

    @property
    def tlwh(self) -> np.ndarray:
        return np.array([self.x, self.y, self.w, self.h], dtype=np.float32)


@dataclass
class GroundTruth:
    """Single ground-truth annotation from gt/gt.txt."""
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    is_active: int   # 0 = ignore, 1 = active
    cls: int         # class id (7 = pedestrian in MOT17)
    visibility: float

    @property
    def tlbr(self) -> np.ndarray:
        return np.array([self.x, self.y, self.x + self.w, self.y + self.h], dtype=np.float32)

    @property
    def tlwh(self) -> np.ndarray:
        return np.array([self.x, self.y, self.w, self.h], dtype=np.float32)


@dataclass
class SequenceInfo:
    name: str
    img_dir: str
    frame_rate: int
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str


class MOT17Sequence:
    """
    Loader for a single MOT17 sequence directory.

    Usage:
        seq = MOT17Sequence("data/MOT17 2/train/MOT17-02-DPM")
        for frame_id, img, dets, gts in seq.iter_frames():
            ...
    """

    def __init__(self, seq_path: str | Path) -> None:
        self.seq_path = Path(seq_path)
        self.info = self._load_seqinfo()
        self._gt: Optional[Dict[int, List[GroundTruth]]] = None
        self._det: Optional[Dict[int, List[Detection]]] = None

    # Public API

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def num_frames(self) -> int:
        return self.info.seq_length

    @property
    def fps(self) -> int:
        return self.info.frame_rate

    def get_frame(self, frame_id: int) -> np.ndarray:
        """Load a single frame (1-indexed) as BGR numpy array."""
        img_path = self.seq_path / self.info.img_dir / f"{frame_id:06d}{self.info.im_ext}"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {img_path}")
        return img

    def get_detections(self, frame_id: int) -> List[Detection]:
        """Return pre-computed detections for the given frame."""
        if self._det is None:
            self._det = self._load_det()
        return self._det.get(frame_id, [])

    def get_ground_truth(self, frame_id: int, active_only: bool = True) -> List[GroundTruth]:
        """Return ground-truth annotations for the given frame."""
        if self._gt is None:
            self._gt = self._load_gt()
        rows = self._gt.get(frame_id, [])
        if active_only:
            rows = [r for r in rows if r.is_active == 1]
        return rows

    def iter_frames(
        self,
        start: int = 1,
        end: Optional[int] = None,
    ) -> Generator[Tuple[int, np.ndarray, List[Detection], List[GroundTruth]], None, None]:
        """
        Iterate over frames, yielding (frame_id, image, detections, ground_truths).

        Args:
            start: first frame (1-indexed, inclusive)
            end:   last frame (inclusive); defaults to seq_length
        """
        end = end or self.info.seq_length
        for fid in range(start, end + 1):
            img = self.get_frame(fid)
            dets = self.get_detections(fid)
            gts = self.get_ground_truth(fid)
            yield fid, img, dets, gts

    # Internal helpers

    def _load_seqinfo(self) -> SequenceInfo:
        ini_path = self.seq_path / "seqinfo.ini"
        cfg = configparser.ConfigParser()
        cfg.read(str(ini_path))
        s = cfg["Sequence"]
        return SequenceInfo(
            name=s["name"],
            img_dir=s.get("imDir", "img1"),
            frame_rate=int(s["frameRate"]),
            seq_length=int(s["seqLength"]),
            im_width=int(s["imWidth"]),
            im_height=int(s["imHeight"]),
            im_ext=s.get("imExt", ".jpg"),
        )

    def _load_gt(self) -> Dict[int, List[GroundTruth]]:
        gt_path = self.seq_path / "gt" / "gt.txt"
        result: Dict[int, List[GroundTruth]] = {}
        if not gt_path.exists():
            return result
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9:
                    continue
                frame = int(parts[0])
                gt = GroundTruth(
                    frame=frame,
                    track_id=int(parts[1]),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    w=float(parts[4]),
                    h=float(parts[5]),
                    is_active=int(parts[6]),
                    cls=int(parts[7]),
                    visibility=float(parts[8]),
                )
                result.setdefault(frame, []).append(gt)
        return result

    def _load_det(self) -> Dict[int, List[Detection]]:
        det_path = self.seq_path / "det" / "det.txt"
        result: Dict[int, List[Detection]] = {}
        if not det_path.exists():
            return result
        with open(det_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                frame = int(parts[0])
                det = Detection(
                    frame=frame,
                    x=float(parts[2]),
                    y=float(parts[3]),
                    w=float(parts[4]),
                    h=float(parts[5]),
                    confidence=float(parts[6]),
                )
                result.setdefault(frame, []).append(det)
        return result


class MOT17Dataset:
    """
    Loader for the entire MOT17 split (train or test).

    Usage:
        dataset = MOT17Dataset("data/MOT17 2", split="train")
        for seq in dataset.sequences:
            print(seq.name, seq.num_frames)
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.sequences: List[MOT17Sequence] = self._discover_sequences()

    def _discover_sequences(self) -> List[MOT17Sequence]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        seqs = sorted(
            [MOT17Sequence(p) for p in split_dir.iterdir() if p.is_dir()],
            key=lambda s: s.name,
        )
        return seqs

    def get_sequence(self, name: str) -> MOT17Sequence:
        for seq in self.sequences:
            if seq.name == name:
                return seq
        raise KeyError(f"Sequence '{name}' not found in {self.split} split")

    def __len__(self) -> int:
        return len(self.sequences)

    def __repr__(self) -> str:
        return f"MOT17Dataset(split={self.split!r}, sequences={len(self)})"
