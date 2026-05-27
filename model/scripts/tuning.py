"""Fine-tune the trained drop detector on a user's own motion patterns.

Usage:
    python tuning.py                       # mix user_data with original train, overwrite model.json
    python tuning.py --keep_old            # write to model_tuned.json instead of overwriting
    python tuning.py --epochs 8 --lr 0.005 # custom hyperparameters
    python tuning.py --user-data my_data/  # custom user-data folder

Expected user_data/ layout (same as drop_detect/):
    user_data/
        idle/      *.csv     (optional)
        walking/   *.csv     (optional)
        fidget/    *.csv     (optional)
        drop/      *.csv     (optional)

Any missing class folder is silently skipped — fine-tune only the classes the user
provided samples for. Normalization stats from the original TRAIN.bin are reused
(so on-device FEATURE_MEAN / FEATURE_INV_STD constants don't change).
"""

import argparse
import json
import os
import struct
import sys
import time
import numpy as np

# Reuse the data.py helpers so feature math stays in lockstep with training.
import data as data_module


CLASS_NAMES = ["idle", "walking", "fidget", "drop"]
NUM_CLASSES = 4
CHANNELS = 4
KERNEL = 7
STRIDE = 2
CONV_OUT = 8


# ---------- I/O ----------

def load_train_bin(path):
    """Returns (X, y, mean, std). X shape (N, T, F), y shape (N,), mean/std shape (F,)."""
    with open(path, "rb") as f:
        N, T, F = struct.unpack("iii", f.read(12))
        mean = np.frombuffer(f.read(4 * F), dtype=np.float32).copy()
        std  = np.frombuffer(f.read(4 * F), dtype=np.float32).copy()
        X = np.frombuffer(f.read(4 * N * T * F), dtype=np.float32).reshape(N, T, F).copy()
        y = np.frombuffer(f.read(4 * N), dtype=np.int32).copy()
    return X, y, mean, std


def load_model_json(path):
    """Reads conv1d/dense weights and biases from model.json."""
    with open(path, "r") as f:
        m = json.load(f)
    conv_w = np.array(m["conv1d"]["weight"], dtype=np.float32).reshape(m["conv1d"]["weight_shape"])
    conv_b = np.array(m["conv1d"]["bias"],   dtype=np.float32)
    dense_w = np.array(m["dense"]["weight"], dtype=np.float32).reshape(m["dense"]["weight_shape"])
    dense_b = np.array(m["dense"]["bias"],   dtype=np.float32)
    return conv_w, conv_b, dense_w, dense_b


def save_model_json(path, conv_w, conv_b, dense_w, dense_b):
    """Writes back in the same shape main.cpp's export_model_json emits."""
    def floats(arr):
        return [float(x) for x in arr.flatten()]
    out = {
        "conv1d": {
            "weight_shape": list(conv_w.shape),
            "weight": floats(conv_w),
            "bias_shape": list(conv_b.shape),
            "bias": floats(conv_b),
        },
        "dense": {
            "weight_shape": list(dense_w.shape),
            "weight": floats(dense_w),
            "bias_shape": list(dense_b.shape),
            "bias": floats(dense_b),
        },
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# ---------- user_data loader ----------

def load_user_data(folder, mean, std):
    """Mirrors data.py's per-file window extraction, then applies the existing normalization."""
    files_by_class = {}
    if not os.path.isdir(folder):
        return np.zeros((0, 249, CHANNELS), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    for cls_name, cls_idx in data_module.label_map.items():
        sub = os.path.join(folder, cls_name)
        if not os.path.isdir(sub):
            continue
        csvs = [os.path.join(sub, f) for f in sorted(os.listdir(sub)) if f.endswith(".csv")]
        if csvs:
            files_by_class[cls_idx] = csvs

    X, y = [], []
    per_class_counts = {}
    for cls_idx, csvs in files_by_class.items():
        n_before = len(X)
        for path in csvs:
            for window in data_module._windows_from_file(path):
                X.append(window)
                y.append(cls_idx)
        per_class_counts[cls_idx] = len(X) - n_before

    if not X:
        return np.zeros((0, 249, CHANNELS), dtype=np.float32), np.zeros((0,), dtype=np.int64), per_class_counts

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # Apply the existing normalization stats from TRAIN.bin — do NOT refit.
    X = ((X - mean) / std).astype(np.float32)
    return X, y, per_class_counts


# ---------- numpy forward / backward ----------
# Architecture must match build_model() in main.cpp exactly.

def conv1d_forward(x, w, b, stride):
    """x: (C, T). w: (OC, IC, K). b: (OC,). Returns out: (OC, L_out)."""
    OC, IC, K = w.shape
    T = x.shape[1]
    L = (T - K) // stride + 1
    out = np.empty((OC, L), dtype=np.float32)
    for oc in range(OC):
        # vectorized over positions: for each window position p, dot input slice with w[oc]
        bias = b[oc]
        for p in range(L):
            t0 = p * stride
            out[oc, p] = (w[oc] * x[:, t0:t0+K]).sum() + bias
    return out


def conv1d_backward(grad_out, x, w, stride):
    """Returns (d_input, dW, db)."""
    OC, IC, K = w.shape
    T = x.shape[1]
    L = grad_out.shape[1]
    dW = np.zeros_like(w)
    db = grad_out.sum(axis=1)
    d_input = np.zeros_like(x)
    for oc in range(OC):
        for p in range(L):
            t0 = p * stride
            g = grad_out[oc, p]
            dW[oc] += g * x[:, t0:t0+K]
            d_input[:, t0:t0+K] += w[oc] * g
    return d_input, dW, db


def relu_forward(x):
    return np.maximum(x, 0)


def relu_backward(grad_out, x):
    g = grad_out.copy()
    g[x < 0] = 0
    return g


def global_max_pool_forward(x):
    """x: (C, T) -> (C,). Returns (out, argmax) where argmax[c] = winning timestep."""
    argmax = np.argmax(x, axis=1)
    out = x[np.arange(x.shape[0]), argmax]
    return out, argmax


def global_max_pool_backward(grad_out, argmax, in_shape):
    g = np.zeros(in_shape, dtype=np.float32)
    g[np.arange(in_shape[0]), argmax] = grad_out
    return g


def dense_forward(x, w, b):
    return w @ x + b


def dense_backward(grad_out, x, w):
    dW = np.outer(grad_out, x)
    db = grad_out
    d_input = w.T @ grad_out
    return d_input, dW, db


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def forward_full(x, conv_w, conv_b, dense_w, dense_b):
    """Returns (probs, cache) where cache has tensors needed for backward."""
    conv_z = conv1d_forward(x, conv_w, conv_b, STRIDE)
    relu_z = relu_forward(conv_z)
    pool_z, argmax = global_max_pool_forward(relu_z)
    logits = dense_forward(pool_z, dense_w, dense_b)
    probs = softmax(logits)
    cache = (x, conv_z, relu_z, argmax, pool_z, logits, probs)
    return probs, cache


def backward_full(cache, target, conv_w, dense_w):
    """Fused softmax+CE gradient: dL/dlogits = probs - target."""
    x, conv_z, relu_z, argmax, pool_z, logits, probs = cache
    grad_logits = probs - target  # (NUM_CLASSES,)

    d_pool, dense_dW, dense_db = dense_backward(grad_logits, pool_z, dense_w)
    d_relu = global_max_pool_backward(d_pool, argmax, relu_z.shape)
    d_conv = relu_backward(d_relu, conv_z)
    _, conv_dW, conv_db = conv1d_backward(d_conv, x, conv_w, STRIDE)

    return conv_dW, conv_db, dense_dW, dense_db


def predict_class(x, conv_w, conv_b, dense_w, dense_b):
    probs, _ = forward_full(x, conv_w, conv_b, dense_w, dense_b)
    return int(np.argmax(probs))


# ---------- evaluation helpers ----------

def evaluate(X, y, conv_w, conv_b, dense_w, dense_b):
    """Returns (overall_acc, per_class_correct, per_class_total)."""
    if len(X) == 0:
        return 0.0, [0]*NUM_CLASSES, [0]*NUM_CLASSES
    per_correct = [0] * NUM_CLASSES
    per_total = [0] * NUM_CLASSES
    correct = 0
    for i in range(len(X)):
        x = X[i].T  # (T,F) -> (C=F, T)
        pred = predict_class(x.astype(np.float32), conv_w, conv_b, dense_w, dense_b)
        per_total[y[i]] += 1
        if pred == y[i]:
            correct += 1
            per_correct[y[i]] += 1
    return correct / len(X), per_correct, per_total


def fmt_acc_line(label, acc, per_correct, per_total):
    parts = []
    for c in range(NUM_CLASSES):
        if per_total[c] > 0:
            parts.append(f"{CLASS_NAMES[c]}={100*per_correct[c]/per_total[c]:.0f}%")
        else:
            parts.append(f"{CLASS_NAMES[c]}=-")
    return f"{label}: {100*acc:.1f}%  [{' '.join(parts)}]"


# ---------- training loop ----------

def fine_tune(X, y, conv_w, conv_b, dense_w, dense_b, epochs, lr, eval_X, eval_y, seed=42):
    """SGD per-sample. Prints live progress per epoch."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))

    print(f"\nFine-tuning: {len(X)} training windows, {epochs} epochs, lr={lr}")
    print("-" * 78)

    # In a non-tty stdout (GUI / pipe), \r progress lines look terrible — skip them.
    show_in_epoch_progress = sys.stdout.isatty()

    pre_acc, pre_pc, pre_pt = evaluate(eval_X, eval_y, conv_w, conv_b, dense_w, dense_b)
    print(fmt_acc_line("Eval BEFORE", pre_acc, pre_pc, pre_pt))
    print()

    for epoch in range(epochs):
        rng.shuffle(idx)
        total_loss = 0.0
        correct = 0
        per_correct = [0]*NUM_CLASSES
        per_total = [0]*NUM_CLASSES

        t_start = time.time()
        for step, si in enumerate(idx):
            x = X[si].T.astype(np.float32)  # (C, T)
            target = np.zeros(NUM_CLASSES, dtype=np.float32)
            target[y[si]] = 1.0

            probs, cache = forward_full(x, conv_w, conv_b, dense_w, dense_b)
            loss = -np.log(probs[y[si]] + 1e-9)
            total_loss += float(loss)

            pred = int(np.argmax(probs))
            per_total[y[si]] += 1
            if pred == y[si]:
                correct += 1
                per_correct[y[si]] += 1

            conv_dW, conv_db, dense_dW, dense_db = backward_full(cache, target, conv_w, dense_w)
            conv_w  -= lr * conv_dW
            conv_b  -= lr * conv_db
            dense_w -= lr * dense_dW
            dense_b -= lr * dense_db

            # Live in-epoch progress every 200 steps so the user sees motion (TTY only).
            if show_in_epoch_progress and ((step + 1) % 200 == 0 or step == len(idx) - 1):
                pct = 100 * (step + 1) / len(idx)
                sys.stdout.write(f"\r  epoch {epoch+1:2d}  {step+1:5d}/{len(idx):<5d} ({pct:3.0f}%)  "
                                 f"running_loss={total_loss/(step+1):.4f}")
                sys.stdout.flush()

        if show_in_epoch_progress:
            sys.stdout.write("\r" + " " * 78 + "\r")
        avg_loss = total_loss / len(idx)
        dt = time.time() - t_start
        acc_line = fmt_acc_line(f"Epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.4f}  ({dt:.1f}s)",
                                correct/len(idx), per_correct, per_total)
        print(acc_line)

    post_acc, post_pc, post_pt = evaluate(eval_X, eval_y, conv_w, conv_b, dense_w, dense_b)
    print()
    print(fmt_acc_line("Eval AFTER ", post_acc, post_pc, post_pt))
    delta = post_acc - pre_acc
    sign = "+" if delta >= 0 else ""
    print(f"Delta on eval set: {sign}{100*delta:.1f}%")
    print("-" * 78)
    return conv_w, conv_b, dense_w, dense_b


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-data", default="user_data", help="folder with idle/walking/fidget/drop subdirs")
    ap.add_argument("--model",     default="model.json", help="input weights")
    ap.add_argument("--train-bin", default="TRAIN.bin",  help="original training set (for mixing + normalization stats)")
    ap.add_argument("--test-bin",  default="TEST.bin",   help="evaluation set used for BEFORE/AFTER readout")
    ap.add_argument("--epochs",    type=int, default=5)
    ap.add_argument("--lr",        type=float, default=0.005)
    ap.add_argument("--keep_old",  action="store_true",
                    help="if set, write tuned weights to model_tuned.json instead of overwriting model.json")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: {args.model} not found. Train first: ./train.exe", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.train_bin):
        print(f"ERROR: {args.train_bin} not found. Run `python data.py` first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.user_data):
        print(f"ERROR: {args.user_data}/ not found. Create it and add CSVs under class subfolders.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.model} ...")
    conv_w, conv_b, dense_w, dense_b = load_model_json(args.model)
    print(f"  conv1d {conv_w.shape}  dense {dense_w.shape}")

    print(f"Loading {args.train_bin} ...")
    X_orig, y_orig, mean, std = load_train_bin(args.train_bin)
    print(f"  N={len(X_orig)}  mean[0]={mean[0]:.4g}  std[0]={std[0]:.4g}")

    print(f"Loading user data from {args.user_data}/ ...")
    result = load_user_data(args.user_data, mean, std)
    if len(result) == 3:
        X_user, y_user, per_cls = result
    else:
        X_user, y_user = result
        per_cls = {}
    if len(X_user) == 0:
        print("ERROR: no CSV windows found in user_data/. Put files under "
              "user_data/<class>/*.csv (class = idle, walking, fidget, drop).", file=sys.stderr)
        sys.exit(1)
    print(f"  User windows: {len(X_user)}  " +
          "  ".join(f"{CLASS_NAMES[k]}={v}" for k, v in per_cls.items()))

    # Mix: concatenate; per-epoch shuffle handles the rest.
    X_mix = np.concatenate([X_orig, X_user], axis=0)
    y_mix = np.concatenate([y_orig, y_user], axis=0)
    print(f"Mixed training set: {len(X_mix)} windows "
          f"({len(X_orig)} original + {len(X_user)} user)")

    # Optional eval set — only used to print BEFORE/AFTER. Falls back to user windows if no test bin.
    if os.path.isfile(args.test_bin):
        X_eval, y_eval, _, _ = load_train_bin(args.test_bin)
    else:
        print(f"  (no {args.test_bin}; using user data as eval set)")
        X_eval, y_eval = X_user, y_user

    conv_w, conv_b, dense_w, dense_b = fine_tune(
        X_mix, y_mix, conv_w, conv_b, dense_w, dense_b,
        epochs=args.epochs, lr=args.lr,
        eval_X=X_eval, eval_y=y_eval,
    )

    out_path = "model_tuned.json" if args.keep_old else args.model
    save_model_json(out_path, conv_w, conv_b, dense_w, dense_b)
    print(f"\nWrote tuned weights to {out_path}")
    if args.keep_old:
        print(f"(original {args.model} preserved)")


if __name__ == "__main__":
    main()
