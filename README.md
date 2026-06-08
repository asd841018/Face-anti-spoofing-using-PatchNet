# Face-anti-spoofing-using-PatchNet

An unofficial PyTorch implementation of
**[PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition](https://arxiv.org/pdf/2203.14325.pdf)** (Wang *et al.*, CVPR 2022).

PatchNet reformulates face anti-spoofing (FAS) from a binary *live / spoof* problem
into a **fine-grained patch-level recognition** problem, where each class is a
combination of the capturing device and the presentation attack instrument (PAI).
Two regularizers — an **Asymmetric AM-Softmax loss** and a **Self-supervised
Similarity loss** — shape the patch embedding space so the model learns
material/texture cues that generalize across identities.

---

## Method overview

```
                       ┌─────────────────────────────┐
  face image  ──►  two random patch views (t1, t2)    │  (training)
                       └──────────────┬──────────────┘
                                      │
                          ResNet-18 backbone (fc → Identity)
                                      │  512-d embedding
                       ┌──────────────┴──────────────┐
                       ▼                              ▼
          Asymmetric AM-Softmax            Self-supervised Similarity
          (fine-grained class)             (pull t1, t2 of same image)
                       └──────────────┬──────────────┘
                          L = α₁·L_Asym + α₂·L_Sim   (α₁ = α₂ = 1)
```

- **Backbone:** ImageNet-pretrained `ResNet-18`, the classification head replaced by
  `nn.Identity()` so it emits a 512-d embedding.
- **Fine-grained classes (OULU-NPU, Protocol 1): 30 classes**
  - `0–5` &nbsp;→ **live**, one class per capturing device (6 phones)
  - `6–29` → **spoof**, 6 devices × 4 attacks (print1, print2, replay1, replay2)
- **Asymmetric AM-Softmax** (`models/AdMSLoss.py`): scale `s = 30`, a **larger margin
  for live** (`ml = 0.4`) than for spoof (`ms = 0.1`), so live faces form a more
  compact cluster while spoof classes stay looser.
- **Self-supervised Similarity loss**: two augmented views of the *same* image are
  pulled together via the L2 distance of their normalized embeddings
  (`L_Sim = mean‖f^{t1} − f^{t2}‖₂`).
- **Patch strategy**
  - *Train:* two random `160×160` crops per image (the two views).
  - *Test:* 9 uniform patches (a 3×3 grid), each resized to `160×160`; the per-patch
    live probabilities are averaged into one image-level score.

---

## Repository structure

```
.
├── train_patchnet.py        # training / validation entry point
├── lr_scheduler.py          # PolyScheduler (polynomial decay, power=2)
├── utils.py                 # metrics: APCER / BPCER / ACER / AUC
├── models/
│   ├── AdMSLoss.py          # Asymmetric AM-Softmax loss + live-prob predictor
│   └── CDCNs.py             # CDCN / CDCN++ (alternative backbone, unused by default)
└── datasets/
    ├── oulup_dataset.py     # OULU-NPU dataset + uniform 3×3 patch cropping
    ├── lcc_fasd.py          # LCC-FASD dataset (cross-dataset test)
    └── augmentation.py      # extra augmentation utilities
```

---

## Requirements

```bash
pip install torch torchvision albumentations opencv-python numpy scikit-learn tensorboard torchsummary
```

Tested with PyTorch ≥ 1.10. A CUDA GPU is required (the AM-Softmax loss allocates
tensors on `cuda`).

---

## Data preparation

### 1. Align faces
Detect and align faces from **OULU-NPU**, save the crops under a root folder
(default in code: `/mnt/training_dataset/face_dataset/Oulu_align`).

### 2. Build the meta files
The dataset reads JSON index files from `<root>/metas/`:

| split | file |
|-------|------|
| train | `metas/AM_train_item.json` |
| val   | `metas/AM_val_item.json`   |
| test  | `metas/test_item.json`     |

Each file is a dict keyed by a **string index**, where every item has a `path`
(relative to the root) and a fine-grained `label`:

```json
{
  "0": { "path": "train/6_1_01_1_frame0.png", "label": 0 },
  "1": { "path": "train/6_1_01_2_frame0.png", "label": 7 }
}
```

**Label convention:** `0–5` are live, `6–29` are spoof. The binary live/spoof flag is
derived automatically as `int(label > 5)`. If your protocol has a different number of
live classes, set `num_live` when constructing `AdMSoftmaxLoss`.

> The hard-coded dataset root and `device = [0, 1]` lines in `train_patchnet.py`
> should be edited to match your environment.

---

## Training

```bash
python train_patchnet.py \
    --lr 0.01 \
    --batchsize 64 \
    --epochs 200 \
    --log CDCNpp_patch
```

| arg | default | description |
|-----|---------|-------------|
| `--lr`           | `0.01`          | initial learning rate (SGD, momentum 0.9, wd 5e-4) |
| `--batchsize`    | `64`            | training batch size |
| `--epochs`       | `200`           | total epochs |
| `--echo_batches` | `50`            | logging interval (in mini-batches) |
| `--log`          | `CDCNpp_patch`  | log directory / run name |

- LR follows a polynomial schedule (`PolyScheduler`, power = 2).
- Validation runs every 5 epochs; the best model (lowest ACER) is saved to
  `models/model.pt` (backbone) and `models/AM_model.pt` (AM-Softmax head).
- TensorBoard logs are written to `./runs`:

  ```bash
  tensorboard --logdir ./runs
  ```

---

## Evaluation & metrics

During validation each image is split into 9 patches; the averaged live probability
is written to `<log>/<log>_map_score_val.txt`, and `utils.performances` reports:

- **APCER** — Attack Presentation Classification Error Rate (attacks accepted as live)
- **BPCER** — Bona fide Presentation Classification Error Rate (live rejected as attack)
- **ACER** — `(APCER + BPCER) / 2`
- **AUC** and threshold-based **ACC**

A cross-dataset test on **LCC-FASD** (`datasets/lcc_fasd.py`) is wired up but
commented out in `train_patchnet.py`; uncomment that block to enable it.

---

## Notes & deviations from the paper

- This is a faithful re-implementation of the core method (fine-grained classes,
  asymmetric margins `0.4 / 0.1`, two-view similarity loss summed over both views).
- The default optimizer here is **SGD**; hyper-parameters (LR, schedule, augmentation)
  are not guaranteed to reproduce the paper's reported numbers exactly.
- `models/CDCNs.py` ships an alternative CDCN/CDCN++ backbone that is **not** used by
  the default pipeline (kept for experimentation).

---

## Citation

```bibtex
@inproceedings{wang2022patchnet,
  title     = {PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition},
  author    = {Wang, Chien-Yi and Lu, Yu-Ding and Yang, Shang-Ta and Lai, Shang-Hong},
  booktitle = {CVPR},
  year      = {2022}
}
```
