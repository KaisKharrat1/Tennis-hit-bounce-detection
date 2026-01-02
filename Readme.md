# 🎾 Tennis Hit & Bounce Detection

This repository contains two complementary approaches (supervised and unsupervised)
to detect **hits** and **bounces** of a tennis ball from ball-tracking data extracted
from video frames of the Roland-Garros 2025 Final.

The input consists of per-frame ball positions `(x, y)` with possible missing detections.
The output is a JSON file identical to the input format, augmented with a predicted action
(`air`, `hit`, or `bounce`) for **every frame**.

---

## 📌 Problem Definition

Given a time series of ball positions extracted from a tennis match video, the goal is to:

- detect ball **hits** (racket-ball contact),
- detect ball **bounces** (ball-ground contact),
- classify all other frames as **air**.

This task is challenging due to:
- missing detections (ball invisibility),
- strong class imbalance (`air` ≫ `hit`, `bounce`),
- fast and subtle motion changes around events.

---

## 🧠 Methods Overview

This repository implements **two different pipelines**:

1. **Supervised learning approach**
2. **Unsupervised rule-based approach**

Both start from the same preprocessing pipeline.

---

## 🔧 Preprocessing

The following preprocessing steps are applied to the raw ball trajectory:

- **Cubic spline interpolation** for short gaps of missing detections
- **Savitzky–Golay smoothing** to reduce noise
- **Feature extraction**:
  - velocity (vx, vy)
  - acceleration (ax, ay)
  - speed and acceleration magnitude
  - angular variation

These features are computed at the **frame level** and used by both methods.

---

## 🟦 Method 1 — Supervised Learning (BiLSTM)

### Model

- **Bidirectional LSTM (BiLSTM)** sequence model
- Frame-level classification with **temporal context**
- Each prediction uses a window of frames **before and after** the target frame

This allows the model to learn the characteristic motion patterns around hits and bounces.

---

### Handling Class Imbalance

Three different strategies were explored to address the strong imbalance between classes:

1. **Cross-Entropy Loss with class weights**  
   Assigns higher importance to minority classes (`hit`, `bounce`).

2. **Focal Loss**  
   Focuses learning on hard-to-classify samples.

3. **Balanced batch sampling**  
   Each training batch contains:
   - 1/3 `air`
   - 1/3 `hit`
   - 1/3 `bounce`

---

### Trade-off and Final Choice

The three approaches exhibit a clear **precision–recall trade-off** for minority classes:

- Weighted losses tend to improve precision but may miss events.
- Balanced batches significantly improve recall for `hit` and `bounce`.

For this project, we **prioritized recall over precision** for `hit` and `bounce`:
> It is preferable to slightly over-detect events than to miss critical hits or bounces.

This choice is motivated by downstream sports analytics use cases, where missing an event
is often more costly than a false positive.

A pretrained BiLSTM checkpoint is provided for inference.

---

## 🟨 Method 2 — Unsupervised Rule-Based Detection

The unsupervised method relies purely on physics-inspired rules derived from ball dynamics.

### Step 1 — Event Candidate Detection

- Compute acceleration magnitude from the smoothed trajectory
- Detect **peaks of acceleration** as potential events
- A frame is considered a valid event **only if the ball is visible**
- All frames where the ball is invisible are labeled as `air`

---

### Step 2 — Serve Detection

- The **serve** is identified as the first event preceded by a period of ball invisibility
- This allows separating the initial hit from rally events

---

### Step 3 — Sequential Event Classification

Detected events are classified using simple temporal constraints:

- Generally, two consecutive `hits` or two consecutive `bounces` are unlikely
- Events tend to alternate in sequence