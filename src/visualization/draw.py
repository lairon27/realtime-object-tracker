"""
Drawing utilities for bounding boxes and track IDs.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Reproducible palette: index → BGR color
_PALETTE_SIZE = 80
_rng = np.random.default_rng(42)
_PALETTE = _rng.integers(80, 255, size=(_PALETTE_SIZE, 3)).tolist()


def _color(idx: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR color for a given integer id."""
    return tuple(_PALETTE[int(idx) % _PALETTE_SIZE])


def draw_detections(
    frame: np.ndarray,
    detections: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_conf: bool = True,
) -> np.ndarray:
    """
    Draw raw detections (no track IDs) on a frame.

    Args:
        frame:      BGR image (H, W, 3). Not modified in-place.
        detections: (N, 6) array — [x1, y1, x2, y2, conf, class_id]
        color:      BGR color for all boxes.
        thickness:  Line thickness.
        show_conf:  Whether to print confidence score.

    Returns:
        Annotated copy of the frame.
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2, conf = int(det[0]), int(det[1]), int(det[2]), int(det[3]), float(det[4])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if show_conf:
            label = f"{conf:.2f}"
            _put_label(out, label, x1, y1, color)
    return out


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence,
    thickness: int = 2,
    show_id: bool = True,
    show_conf: bool = False,
) -> np.ndarray:
    """
    Draw tracked objects with per-ID colors.

    Args:
        frame:   BGR image.
        tracks:  Sequence of objects with attributes:
                   .track_id (int)
                   .tlbr     (np.ndarray [x1, y1, x2, y2])
                   .score    (float, optional)
        show_id:   Draw track ID label.
        show_conf: Draw confidence score next to ID.

    Returns:
        Annotated copy of the frame.
    """
    out = frame.copy()
    for t in tracks:
        tid = int(t.track_id)
        x1, y1, x2, y2 = (int(v) for v in t.tlbr)
        color = _color(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if show_id:
            label = f"ID {tid}"
            if show_conf and hasattr(t, "score"):
                label += f" {t.score:.2f}"
            _put_label(out, label, x1, y1, color)
    return out


def draw_mot17_gt(
    frame: np.ndarray,
    ground_truths,
    color: Tuple[int, int, int] = (255, 128, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw MOT17 GroundTruth annotations (from data_loader)."""
    out = frame.copy()
    for gt in ground_truths:
        x1, y1, x2, y2 = (int(v) for v in gt.tlbr)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        _put_label(out, f"GT {gt.track_id}", x1, y1, color)
    return out


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 4) -> np.ndarray:
    """Concatenate two same-height images horizontally with a thin gap."""
    h = max(left.shape[0], right.shape[0])

    def _pad(img):
        if img.shape[0] < h:
            pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
            img = np.vstack([img, pad])
        return img

    divider = np.zeros((h, gap, 3), dtype=left.dtype)
    return np.hstack([_pad(left), divider, _pad(right)])


def resize_for_display(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """Downscale frame to at most max_width, keeping aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# Internal

def _put_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw a filled-background text label just above (x, y)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    ty = max(y - 4, th + 4)
    cv2.rectangle(img, (x, ty - th - baseline - 2), (x + tw, ty + baseline), color, cv2.FILLED)
    cv2.putText(img, text, (x, ty - baseline), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
