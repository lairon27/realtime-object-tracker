# Data

This directory contains the MOT17 dataset. **Data files are not committed to git.**

## Download MOT17

1. Download **MOT17** (~5 GB) from https://motchallenge.net/data/MOT17/
2. Extract to this directory so the structure looks like:

```
data/
└── MOT17/
    ├── train/
    │   ├── MOT17-02-DPM/
    │   │   ├── det/det.txt
    │   │   ├── gt/gt.txt
    │   │   ├── img1/000001.jpg ...
    │   │   └── seqinfo.ini
    │   └── ...
    └── test/
        └── ...
```

## Quick check

```python
from src.data_loader import MOT17Sequence
seq = MOT17Sequence("data/MOT17/train/MOT17-02-DPM")
img = seq.get_frame(1)
print(img.shape)  # (1080, 1920, 3)
```
