"""
Appearance feature extractor for Re-ID in DeepSORT.

Architecture: ResNet18 backbone (pretrained on ImageNet), global average
pooling output (512-dim) projected to 128-dim, then L2-normalised.
This gives a compact appearance descriptor for each detected person crop.

Cosine distance between two descriptors measures appearance similarity:
  dist = 1 - (a · b)   (since both are unit vectors)
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class AppearanceExtractor:
    """
    Extract L2-normalised 128-dim appearance embeddings from image crops.

    Args:
        device:    'cpu', 'cuda', or 'mps'.
        embed_dim: Output embedding dimension (default 128).
    """

    _INPUT_SIZE = (128, 64)  # (H, W) — standard Re-ID input: tall crop

    def __init__(
        self,
        device: str = "cpu",
        embed_dim: int = 128,
    ) -> None:
        self.device    = torch.device(device)
        self.embed_dim = embed_dim
        self.model     = self._build_model(embed_dim).to(self.device).eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

     # Public API
    
    def extract(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Extract embeddings for a batch of bounding boxes from one frame.

        Args:
            frame: BGR image (H, W, 3).
            boxes: (N, 4) array of [x1, y1, x2, y2] in pixel coords.

        Returns:
            embeddings: (N, embed_dim) float32 array, each row is L2-normalised.
                        Returns (0, embed_dim) array when boxes is empty.
        """
        if len(boxes) == 0:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        crops = self._crop_batch(frame, boxes)      # list of RGB PIL-like arrays
        tensor = self._preprocess(crops)            # (N, 3, H, W)

        with torch.no_grad():
            emb = self.model(tensor.to(self.device))  # (N, embed_dim)

        return emb.cpu().numpy().astype(np.float32)

    def extract_single(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Extract embedding for a single [x1, y1, x2, y2] box."""
        return self.extract(frame, box[None])[0]

        # Internal
    
    def _crop_batch(self, frame: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """Crop and resize each bbox; return list of RGB uint8 arrays."""
        h, w = frame.shape[:2]
        crops = []
        for box in boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]))
            y2 = min(h, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                # Degenerate box → black patch
                crop = np.zeros((*self._INPUT_SIZE, 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (self._INPUT_SIZE[1], self._INPUT_SIZE[0]),
                                  interpolation=cv2.INTER_LINEAR)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)
        return crops

    def _preprocess(self, crops: List[np.ndarray]) -> torch.Tensor:
        tensors = [self.transform(c) for c in crops]
        return torch.stack(tensors)  # (N, 3, H, W)

    @staticmethod
    def _build_model(embed_dim: int) -> nn.Module:
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace final FC: 512 → embed_dim, then L2 normalise
        backbone.fc = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            L2Norm(),
        )
        return backbone


class L2Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, p=2, dim=1)

 
# Distance utilities

def cosine_distance(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine distances between two sets of L2-normalised embeddings.

    Args:
        embeddings_a: (M, D) — e.g. gallery embeddings per track
        embeddings_b: (N, D) — e.g. current frame embeddings

    Returns:
        distance matrix (M, N), values in [0, 2].
        0 = identical direction, 1 = orthogonal, 2 = opposite.
    """
    sim = embeddings_a @ embeddings_b.T   # (M, N)
    return (1.0 - sim).astype(np.float32)


def nearest_cosine_distance(
    gallery: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """
    For each query embedding, find the minimum cosine distance to any
    embedding in the gallery (track's appearance memory).

    Args:
        gallery: (K, D) — stored embeddings for one track
        query:   (N, D) — current frame embeddings

    Returns:
        (N,) minimum cosine distances
    """
    if len(gallery) == 0:
        return np.full(len(query), 1.0, dtype=np.float32)
    dist = cosine_distance(gallery, query)   # (K, N)
    return dist.min(axis=0)                  # (N,)


def appearance_cost_matrix(
    track_galleries: List[np.ndarray],
    det_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Build (M, N) appearance cost matrix.

    Args:
        track_galleries: list of M arrays, each shape (K_i, D)
        det_embeddings:  (N, D)

    Returns:
        cost: (M, N) float32
    """
    M = len(track_galleries)
    N = len(det_embeddings)
    if M == 0 or N == 0:
        return np.empty((M, N), dtype=np.float32)

    cost = np.zeros((M, N), dtype=np.float32)
    for i, gallery in enumerate(track_galleries):
        cost[i] = nearest_cosine_distance(gallery, det_embeddings)
    return cost
