import os
import csv
import numpy as np


label_map = {
    "idle": 0,
    "walking": 1,
    "fidget": 2,
    "drop": 3,
}

WINDOW = 250
STRIDE = 125
FEATURES = 4
TEST_RATIO = 0.2
SEED = 42


# Features are squared magnitudes and their first differences.
# Squared form is deliberate: the Flipper recomputes these on every IMU sample
# inside a live 2.5s window, and sqrt on Cortex-M4 is too slow.
# "a - a_prev" is the delta of the squared magnitude, not true jerk.

def a_mag_sq(ax, ay, az):
    return ax*ax + ay*ay + az*az


def g_mag_sq(gx, gy, gz):
    return gx*gx + gy*gy + gz*gz


# ---------- window extraction ----------

def _windows_from_file(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or not rows[0]:
        return []

    if rows[0][0].lower() == "order":
        rows = rows[1:]

    if len(rows) < WINDOW:
        return []

    out = []
    for start in range(0, len(rows) - WINDOW + 1, STRIDE):
        window = rows[start:start + WINDOW]
        sample = []
        # Loop starts at t=1 because feat 2,3 need the previous timestep.
        # 250-row window produces 249 feature vectors.
        for t in range(1, WINDOW):
            ax, ay, az = map(float, window[t][1:4])
            gx, gy, gz = map(float, window[t][4:7])
            axp, ayp, azp = map(float, window[t-1][1:4])
            gxp, gyp, gzp = map(float, window[t-1][4:7])

            a = a_mag_sq(ax, ay, az)
            g = g_mag_sq(gx, gy, gz)
            a_prev = a_mag_sq(axp, ayp, azp)
            g_prev = g_mag_sq(gxp, gyp, gzp)

            sample.append([a, g, a - a_prev, g - g_prev])

        out.append(sample)
    return out


# ---------- split ----------

def _files_per_class(base_folder):
    """Returns {label: [csv_path, ...]} for every class folder that exists."""
    out = {}
    for label_name, label in label_map.items():
        folder = os.path.join(base_folder, label_name)
        if not os.path.isdir(folder):
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
        files.sort()
        out[label] = files
    return out


def _split_files(files_by_class, test_ratio, seed):
    """Stratified per-class file-level split. Whole files go to train OR test, never both."""
    rng = np.random.default_rng(seed)
    train, test = [], [] 
    for label, files in files_by_class.items():
        idx = np.arange(len(files))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(files) * test_ratio))) if len(files) > 1 else 0
        n_train = len(files) - n_test
        for i in idx[:n_train]:
            train.append((files[i], label))
        for i in idx[n_train:]:
            test.append((files[i], label))
    return train, test


def _build_from_file_list(file_list):
    X, y = [], []
    for path, label in file_list:
        for window in _windows_from_file(path):
            X.append(window)
            y.append(label)
    if not X:
        return np.zeros((0, WINDOW - 1, FEATURES), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------- normalization ----------

def _fit_normalization(X_train):
    """Per-feature mean/std over (N, T) flattened. Returns mean[F], std[F]."""
    # X_train shape: (N, T, F). Reshape to (N*T, F).
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    # Guard against zero-variance features.
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _apply_normalization(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# ---------- entry points ----------

def fetch_data(test_ratio=TEST_RATIO, seed=SEED):
    files_by_class = _files_per_class("drop_detect")
    if not files_by_class:
        raise RuntimeError("No class folders found under drop_detect/")

    train_files, test_files = _split_files(files_by_class, test_ratio, seed)
    X_train, y_train = _build_from_file_list(train_files)
    X_test, y_test = _build_from_file_list(test_files)

    if X_train.size == 0:
        raise RuntimeError("Train set is empty after split.")

    mean, std = _fit_normalization(X_train)
    X_train = _apply_normalization(X_train, mean, std)
    if X_test.size > 0:
        X_test = _apply_normalization(X_test, mean, std)

    return X_train, X_test, y_train, y_test, mean, std


# ---------- binary writer ----------

def save_bin(path, X, y, mean, std):
    # Header: [N, T, F] int32; mean[F] float32; std[F] float32; X float32; y int32.
    N, T, F = X.shape
    with open(path, "wb") as f:
        np.array([N, T, F], dtype=np.int32).tofile(f)
        mean.astype(np.float32).tofile(f)
        std.astype(np.float32).tofile(f)
        X.astype(np.float32).tofile(f)
        y.astype(np.int32).tofile(f)


def _print_class_histogram(name, y):
    counts = np.bincount(y, minlength=len(label_map))
    parts = [f"{lname}={counts[lbl]}" for lname, lbl in label_map.items()]
    print(f"  {name}: N={len(y)}  " + "  ".join(parts))


def _print_normalization_stats(mean, std):
    inv_std = (1.0 / std).astype(np.float32)
    feat_names = ["a_mag_sq", "g_mag_sq", "a_mag_sq_delta", "g_mag_sq_delta"]
    print()
    print("// Per-feature normalization (" + ", ".join(feat_names) + ")")
    print("static const float FEATURE_MEAN[4]    = { " +
          ", ".join(f"{m:.9g}f" for m in mean) + " };")
    print("static const float FEATURE_INV_STD[4] = { " +
          ", ".join(f"{v:.9g}f" for v in inv_std) + " };")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, mean, std = fetch_data()
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    _print_class_histogram("train", y_train)
    _print_class_histogram("test ", y_test)
    save_bin("TRAIN.bin", X_train, y_train, mean, std)
    save_bin("TEST.bin", X_test, y_test, mean, std)
    print(f"Saved TRAIN.bin: {X_train.shape}, TEST.bin: {X_test.shape}")
    _print_normalization_stats(mean, std)
