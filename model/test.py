import os, csv

label_map = {"idle": 0, "walking": 1, "fidget": 2, "drop": 3}
WINDOW = 250

for label_name in label_map:
    folder = os.path.join("drop_detect", label_name)
    if not os.path.isdir(folder):
        print(f"{label_name}: folder missing")
        continue
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    kept = 0
    for file in files:
        with open(os.path.join(folder, file)) as f:
            rows = list(csv.reader(f))
        if rows and rows[0][0].lower() == "order":
            rows = rows[1:]
        if len(rows) >= WINDOW:
            kept += 1
        else:
            print(f"  {label_name}/{file}: only {len(rows)} rows (need {WINDOW})")
    print(f"{label_name}: {kept}/{len(files)} files passed")