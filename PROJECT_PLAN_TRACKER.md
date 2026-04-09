# Real-time Object Tracking Pipeline

## Мета проекту

Побудувати end-to-end pipeline для детекції та трекінгу об'єктів у відео в реальному часі.
Показати розуміння detection, multi-object tracking, data association та re-identification.

---

## Датасет

**MOT17 (Multiple Object Tracking Benchmark 2017)**

### Що скачати

| Файл | Посилання |
|------|-----------|
| MOT17 Train | https://motchallenge.net/data/MOT17/ |
| MOT17 Test | https://motchallenge.net/data/MOT17/ |

Датасет містить відео з пішоходами, bounding box анотації, detection результати від різних детекторів.
Розмір: ~5 GB. Потрібна реєстрація на motchallenge.net.

### Структура датасету

```
MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   │   ├── det/           # готові детекції
│   │   ├── gt/            # ground truth
│   │   ├── img1/          # кадри як .jpg
│   │   └── seqinfo.ini    # метадані (fps, resolution)
│   ├── MOT17-04-DPM/
│   └── ...
└── test/
```

---

## Репозиторій

**Назва:** `realtime-object-tracker`

**Description:**
> Real-time multi-object tracking pipeline with YOLOv8 detection and ByteTrack/DeepSORT association. Evaluated on MOT17 benchmark with MOTA/IDF1 metrics. Includes Streamlit demo.

**Topics:** `object-tracking`, `computer-vision`, `yolov8`, `deepsort`, `bytetrack`, `mot`, `python`, `real-time`

---

## Структура репозиторію

```
realtime-object-tracker/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml              # параметри трекера, детектора, шляхи
├── data/
│   └── README.md                 # інструкції де скачати MOT17 (самі дані НЕ комітити)
├── src/
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py           # YOLOv8 обгортка (ultralytics)
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── bytetrack.py          # ByteTrack алгоритм
│   │   ├── deepsort.py           # DeepSORT алгоритм
│   │   └── kalman_filter.py      # Kalman filter для prediction
│   ├── association/
│   │   ├── __init__.py
│   │   ├── iou_matching.py       # IoU-based matching
│   │   ├── hungarian.py          # Hungarian algorithm
│   │   └── appearance.py         # Re-ID feature extractor
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py            # MOTA, IDF1, ID switches
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── tracker.py            # end-to-end: відео → detection → tracking → output
│   └── visualization/
│       ├── __init__.py
│       └── draw.py               # малювати bbox + track ID на кадрах
├── notebooks/
│   └── demo.ipynb
├── scripts/
│   ├── run_tracker.py            # CLI: python scripts/run_tracker.py --input video.mp4
│   ├── evaluate.py               # CLI: порахувати метрики на MOT17
│   └── run_streamlit.py          # Streamlit демо
├── tests/
│   ├── test_kalman.py
│   ├── test_iou.py
│   └── test_hungarian.py
└── assets/
    ├── architecture.png
    └── demo.gif
```

---

## План роботи по етапах

### Етап 1 — Підготовка (1 день)

- [ ] Створити репо на GitHub
- [ ] Скачати MOT17 train set
- [ ] Перевірити що кадри читаються: `cv2.imread("img1/000001.jpg")`
- [ ] Написати `.gitignore`, `requirements.txt`
- [ ] Зробити data loader для MOT17 формату (читати gt.txt, det.txt, кадри)

### Етап 2 — Detection (1-2 дні)

- [ ] Написати обгортку для YOLOv8: вхід — кадр, вихід — список `[x1, y1, x2, y2, confidence, class]`
- [ ] Прогнати на кількох послідовностях MOT17
- [ ] Відфільтрувати тільки клас "person"
- [ ] Візуально перевірити: намалювати bbox на кадрах
- [ ] Порівняти з готовими детекціями з MOT17 (det/det.txt)

### Етап 3 — Kalman Filter (1-2 дні)

- [ ] Реалізувати Kalman Filter для bounding box prediction
- [ ] State: `[x_center, y_center, aspect_ratio, height, vx, vy, va, vh]`
- [ ] Predict: де буде bbox на наступному кадрі
- [ ] Update: скоригувати по реальній детекції
- [ ] Тест: синтетична траєкторія → зашумлені спостереження → перевірити що фільтр згладжує

### Етап 4 — Data Association (2-3 дні)

- [ ] Реалізувати IoU matching: порахувати IoU між predicted та detected bbox
- [ ] Реалізувати Hungarian algorithm для оптимального matching
- [ ] Логіка: matched → update track, unmatched detection → new track, unmatched track → lost/delete
- [ ] Поріг: min IoU для match (зазвичай 0.3)
- [ ] Це вже базовий SORT — протестувати на MOT17

### Етап 5 — ByteTrack (1-2 дні)

- [ ] Реалізувати ByteTrack поверх SORT
- [ ] Ключова ідея: два етапи matching — спочатку high-confidence детекції, потім low-confidence
- [ ] Low-confidence детекції матчити з unmatched tracks (рятує часткові оклюзії)
- [ ] Порівняти результати SORT vs ByteTrack

### Етап 6 — DeepSORT / Appearance (2-3 дні, опціонально)

- [ ] Додати Re-ID feature extractor (простий ResNet або OSNet)
- [ ] Для кожного bbox — витягнути appearance embedding (128-dim вектор)
- [ ] Комбінувати IoU cost + cosine distance для кращого matching
- [ ] Це допомагає коли людина зникає за оклюзією і з'являється знову
- [ ] Порівняти: SORT vs ByteTrack vs DeepSORT

### Етап 7 — Evaluation (1 день)

- [ ] Реалізувати метрики: MOTA, IDF1, ID Switches, FP, FN
- [ ] Або використати бібліотеку `motmetrics` (pip install motmetrics)
- [ ] Прогнати evaluation на MOT17 train sequences
- [ ] Зібрати таблицю результатів для README

### Етап 8 — Демо та оформлення (1-2 дні)

- [ ] Streamlit демо: завантаж відео → бачиш трекінг в реальному часі
- [ ] Або CLI: `python scripts/run_tracker.py --input video.mp4 --output result.mp4`
- [ ] Зробити demo.gif для README
- [ ] README: архітектура, результати, Quick Start, таблиця метрик
- [ ] Docstrings, тести

---

## Технології

| Що | Чим |
|----|-----|
| Мова | Python 3.10+ |
| Detection | ultralytics (YOLOv8) |
| Tracking | власна реалізація SORT / ByteTrack / DeepSORT |
| Лінійна алгебра | NumPy, SciPy (Hungarian: `scipy.optimize.linear_sum_assignment`) |
| Re-ID | torchvision (ResNet) або torchreid |
| Evaluation | motmetrics |
| Демо | Streamlit |
| Тести | pytest |

---

## Ключові алгоритми для розуміння

### Kalman Filter
- Predict → Update цикл
- State estimation під шумом
- Часто питають на інтерв'ю

### Hungarian Algorithm
- Оптимальне призначення (detection → track)
- O(n³), scipy має готову реалізацію
- Треба розуміти cost matrix

### IoU (Intersection over Union)
- Міра перекриття двох bbox
- Основа для matching та evaluation

### ByteTrack vs DeepSORT
- ByteTrack: двоетапне matching по confidence, без appearance features — швидкий і ефективний
- DeepSORT: appearance embedding + cascade matching — краще для re-identification після оклюзій

---

## Що це покаже роботодавцю

1. **Detection + Tracking** — найпопулярніша задача в продакшн CV
2. **Розуміння алгоритмів** — Kalman filter, Hungarian, IoU — все своїми руками
3. **Порівняння методів** — SORT vs ByteTrack vs DeepSORT з метриками
4. **Продакшн мислення** — Streamlit демо, CLI, конфіги, тести
5. **Стандартний бенчмарк** — MOT17 це те що всі знають, результати легко порівняти

---

## Коміт (якщо одним комітом)

```
feat: real-time multi-object tracking pipeline

End-to-end multi-object tracking with YOLOv8 detection and
ByteTrack/DeepSORT association. Evaluated on MOT17 benchmark.

- YOLOv8 detection wrapper with confidence filtering
- Kalman filter for bounding box state prediction
- IoU and appearance-based data association
- Hungarian algorithm for optimal matching
- ByteTrack two-stage matching for occluded objects
- DeepSORT with Re-ID appearance features
- MOTA/IDF1 evaluation on MOT17 train set
- Streamlit demo and CLI interface
- Unit tests for core components
```

---

## Після цього проекту

Портфоліо буде:
1. **multicam-3d-reconstruction** — геометричне CV, калібрація, тріангуляція
2. **realtime-object-tracker** — detection, tracking, data association

Для третього проекту варто щось з deep learning: depth estimation, segmentation, або visual SLAM.
