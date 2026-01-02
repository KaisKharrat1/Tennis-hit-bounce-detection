import json
import numpy as np
import torch
from scipy.signal import find_peaks

from functions import temporal_nms, extract_xy, spline_interpolate, smooth, compute_kinematics
from model import HitBounceBiLSTM


def supervised_hit_bounce_detection(ball_data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HitBounceBiLSTM(input_dim=9).to(device)
    checkpoint_path="experiments/run_01/checkpoints/epoch_25.pt"
    window = 5

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    frames = sorted(ball_data.keys(), key=lambda x: int(x))

    # Extraction + features
        
    t, x_raw, y_raw = extract_xy(ball_data)

    x_i = spline_interpolate(t, x_raw)
    y_i = spline_interpolate(t, y_raw)

    x_s = smooth(x_i)
    y_s = smooth(y_i)

    vx, vy, ax, ay = compute_kinematics(t, x_s, y_s)

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

    # Sliding window
    
    half = window // 2
    N = len(features)

    preds = np.zeros(N, dtype=int)  # défaut = air
    probs = np.zeros((N, 3))

    for i in range(half, N - half):
        X_win = features[i - half:i + half + 1]
        X_win = torch.tensor(X_win, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(X_win)
            p = torch.softmax(logits, dim=1).cpu().numpy()[0]

        probs[i] = p
        preds[i] = np.argmax(p)

    # Post-processing    

    preds = temporal_nms(probs, preds, cls=1)
    preds = temporal_nms(probs, preds, cls=2)

    # Reconstruction JSON

    ID2LABEL = {0: "air", 1: "hit", 2: "bounce"}

    enriched = {}
    for i, f in enumerate(frames):
        enriched[f] = ball_data[f].copy()
        enriched[f]["pred_action"] = ID2LABEL[preds[i]]

    return enriched

def unsupervised_hit_bounce_detection(ball_data):

    # Pretreatment
    frames = sorted(ball_data.keys(), key=lambda x: int(x))
    frames_int = [int(f) for f in frames]

    t, x_raw, y_raw = extract_xy(ball_data)

    x_i = spline_interpolate(t, x_raw, max_gap=10)
    y_i = spline_interpolate(t, y_raw, max_gap=10)

    x_s = smooth(x_i)
    y_s = smooth(y_i)

    vx, vy, ax, ay = compute_kinematics(t, x_s, y_s)

    accel = np.sqrt(ax**2 + ay**2)

    visible = np.array([ball_data[str(f)]["visible"] for f in frames_int])

    # Events detection
    event_idx, _ = find_peaks(
        accel,
        height=4.5,
        distance=17
    )

    
    # Snap to visible frames
    
    def snap_to_visible(i, visible, max_search=10):
        if visible[i]:
            return i
        for d in range(1, max_search + 1):
            if i - d >= 0 and visible[i - d]:
                return i - d
            if i + d < len(visible) and visible[i + d]:
                return i + d
        return None

    cleaned_idx = []
    for i in event_idx:
        j = snap_to_visible(i, visible)
        if j is not None:
            cleaned_idx.append(j)

    cleaned_idx = sorted(set(cleaned_idx))
    event_frames = [frames_int[i] for i in cleaned_idx]

    if len(event_frames) == 0:
        enriched = {}
        for f in frames:
            enriched[f] = ball_data[f].copy()
            enriched[f]["pred_action"] = "air"
        return enriched

    
    # long invisibility before (detection of a serve)
    
    def count_invisible_before(frame, window=10):
        c = 0
        for f in range(frame - 1, frame - window - 1, -1):
            if str(f) not in ball_data:
                break
            if not ball_data[str(f)]["visible"]:
                c += 1
            else:
                break
        return c

    invis_counts = [count_invisible_before(f) for f in event_frames]
    service_event_idx = int(np.argmax(invis_counts))


    # Alternance hit / bounce
    
    event_labels = {}
    event_labels[service_event_idx] = "hit"

    # after the serve
    for i in range(service_event_idx + 1, len(event_frames)):
        prev = event_labels[i - 1]
        event_labels[i] = "bounce" if prev == "hit" else "hit"

    # before the serve
    for i in range(service_event_idx - 1, -1, -1):
        nxt = event_labels[i + 1]
        event_labels[i] = "bounce" if nxt == "hit" else "hit"

    predicted_events = {
        str(event_frames[i]): event_labels[i]
        for i in range(len(event_frames))
    }

    # 6. Reconstruction JSON 
    enriched = {}
    for f in frames:
        enriched[f] = ball_data[f].copy()
        enriched[f]["pred_action"] = predicted_events.get(f, "air")

    return enriched
