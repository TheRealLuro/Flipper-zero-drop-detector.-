import os
import csv
import numpy as np


label_map = {
    "idle": 0,
    "walking": 1,
    "fidgeting": 2,
    "drop": 3
}

WINDOW = 250
FEATURES = 4


# ---------- feature functions ----------

def a_mag(ax, ay, az):
    return ax*ax + ay*ay + az*az


def g_mag(gx, gy, gz):
    return gx*gx + gy*gy + gz*gz


# ---------- main dataset builder ----------

def build_conv1d_dataset(base_folder):

    X = []
    y = []

    for label_name, label in label_map.items():

        folder = os.path.join(base_folder, label_name)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):

            if not file.endswith(".csv"):
                continue

            path = os.path.join(folder, file)

    
            with open(path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)


            if not rows or not rows[0]:
                continue

    
            if rows[0][0].lower() == "order":
                rows = rows[1:]

           
            if len(rows) < WINDOW:
                continue

            rows = rows[:WINDOW]

            
            sample = []

            for t in range(1, WINDOW):

                ax, ay, az = map(float, rows[t][1:4])
                gx, gy, gz = map(float, rows[t][4:7])

                axp, ayp, azp = map(float, rows[t-1][1:4])
                gxp, gyp, gzp = map(float, rows[t-1][4:7])

                a = a_mag(ax, ay, az)
                g = g_mag(gx, gy, gz)

                a_prev = a_mag(axp, ayp, azp)
                g_prev = g_mag(gxp, gyp, gzp)

                ja = a - a_prev
                jg = g - g_prev

                sample.append([a, g, ja, jg])

        
            X.append(sample)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=42):

    rng = np.random.default_rng(seed)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    split = int(len(X) * (1 - test_ratio))

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def fetch_data(test_ratio=0.2):
    X, y = build_conv1d_dataset("drop_detect")
    return train_test_split(X, y, test_ratio=test_ratio)



print(fetch_data())