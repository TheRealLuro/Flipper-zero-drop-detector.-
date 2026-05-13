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

            # ---------- STEP 1: load CSV ----------
            with open(path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # remove header if needed
            # skip empty files
            if not rows or not rows[0]:
                continue

            # skip header safely
            if rows[0][0].lower() == "order":
                rows = rows[1:]

            # need full window
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





def fetch_data():
    # X and y dataset structure for Conv1D model:
#
# X = input data (features)
# Each X[i] is ONE sample representing a full 2.5 second time window.
# Shape of X[i] = (250, 4)
#   - 250 = number of timesteps (100Hz sampling → 2.5 seconds)
#   - 4 = engineered features per timestep:
#       [a_mag^2, g_mag^2, accel_jerk, gyro_jerk]
#
# So X shape overall = (num_samples, 250, 4)
#
# y = labels (targets)
# Each y[i] corresponds to ONE full 2.5 second window in X[i].
# Labels:
#   0 = idle
#   1 = walking
#   2 = fidgeting
#   3 = drop
#
# How it is used:
# The Conv1D model looks at the entire 250-step sequence at once
# and learns patterns over time (not individual rows).
# It outputs a single prediction per window (classification of the event).
#
# Final mapping:
# X[i] (250 timesteps of motion) → model → y[i] (single class label)
    X, y = build_conv1d_dataset("drop_detect")
    return X, y




