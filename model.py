import json
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.utils.data import Sampler
from collections import defaultdict
from torch.utils.data import Dataset

from functions import extract_xy
from functions import spline_interpolate
from functions import smooth
from functions import compute_kinematics

LABEL_MAP = {"air": 0, "hit": 1, "bounce": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def process_json(json_path, window=10):
    with open(json_path, "r") as f:
        ball_data = json.load(f)

    t, x_raw, y_raw = extract_xy(ball_data)
    x_i = spline_interpolate(t, x_raw)
    y_i = spline_interpolate(t, y_raw)
    x_s = smooth(x_i)
    y_s = smooth(y_i)

    vx, vy, ax, ay = compute_kinematics(t, x_s, y_s)

    # --- security : replace NaN and clip ---
    vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
    vy = np.nan_to_num(vy, nan=0.0, posinf=0.0, neginf=0.0)
    ax = np.nan_to_num(ax, nan=0.0, posinf=0.0, neginf=0.0)
    ay = np.nan_to_num(ay, nan=0.0, posinf=0.0, neginf=0.0)
    x_s = np.nan_to_num(x_s, nan=0.0, posinf=0.0, neginf=0.0)
    y_s = np.nan_to_num(y_s, nan=0.0, posinf=0.0, neginf=0.0)

    speed = np.sqrt(vx**2 + vy**2)
    acceleration = np.sqrt(ax**2 + ay**2)
    angle = np.arctan2(vy, vx)
    d_angle = np.gradient(angle)

    features = np.stack([
        x_s, y_s,
        vx, vy,
        ax, ay,
        speed,
        acceleration,
        d_angle
    ], axis=1)

    # --- clip ---
    features = np.clip(features, -1e3, 1e3)

    # --- normalisation  ---
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-6
    features = (features - mean) / std

    # --- verification ---
    if not np.all(np.isfinite(features)):
        raise ValueError(f"NaN ou Inf restant dans {json_path}")

    # --- labels ---
    frames = sorted(ball_data.keys(), key=lambda x: int(x))
    labels = np.array([LABEL_MAP[ball_data[f]["action"]] for f in frames])

    # --- creating windows for the model ---
    X, y = [], []
    half = window // 2
    for i in range(half, len(features) - half):
        X.append(features[i - half:i + half + 1])
        y.append(labels[i])

    return np.array(X), np.array(y)

class TennisDataset(Dataset):
    def __init__(self, json_files, window=10):
        self.samples = []
        self.window = window

        for path in json_files:
            X, y = process_json(path, window)
            for Xi, yi in zip(X, y):
                self.samples.append((
                    torch.tensor(Xi, dtype=torch.float32),
                    torch.tensor(yi, dtype=torch.long)
                ))

    def check_nan(self, X, name="X"):
        if torch.isnan(X).any():
            raise ValueError(f"NaN détecté dans {name}")
        if torch.isinf(X).any():
            raise ValueError(f"Inf détecté dans {name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        Xi, yi = self.samples[idx]
        self.check_nan(Xi, "features")
        return Xi, yi

class BalancedTennisDataset(Dataset):
    def __init__(self, json_files, window=10):
        self.samples = []
        self.labels = [] 
        self.window = window

        for path in json_files:
            X, y = process_json(path, window)
            for Xi, yi in zip(X, y):
                self.samples.append((
                    torch.tensor(Xi, dtype=torch.float32),
                    torch.tensor(yi, dtype=torch.long)
                ))
                self.labels.append(int(yi))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        assert batch_size % 3 == 0
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = batch_size // 3

        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)

        min_class_size = min(len(v) for v in self.class_indices.values())

        self.num_batches = max(
            1,
            min_class_size // self.samples_per_class
        )

        print("Sampler stats:")
        for k in self.class_indices:
            print(f"  Classe {k}: {len(self.class_indices[k])} samples")
        print(f"  Nb batches / epoch: {self.num_batches}")

    def __iter__(self):
        for k in self.class_indices:
            random.shuffle(self.class_indices[k])

        pointers = {k: 0 for k in self.class_indices}

        for _ in range(self.num_batches):
            batch = []
            for cls in [0, 1, 2]:
                start = pointers[cls]
                end = start + self.samples_per_class

                if end > len(self.class_indices[cls]):
                    random.shuffle(self.class_indices[cls])
                    start = 0
                    end = self.samples_per_class
                    pointers[cls] = end
                else:
                    pointers[cls] = end

                batch.extend(self.class_indices[cls][start:end])

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

    
class HitBounceBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x : (B, T, F)
        out, _ = self.lstm(x)
        center = out[:, out.shape[1] // 2, :]  # center frame
        return self.fc(center)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
