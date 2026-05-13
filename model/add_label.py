import os
import csv

label_map = {
    "idle": 0,
    "walking": 1,
    "fidget": 2,
    "drop": 3
}


def append_labels_to_csvs(base_folder):

    for label_name, label_value in label_map.items():

        folder_path = os.path.join(base_folder, label_name)

        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):

            if not filename.endswith(".csv"):
                continue

            file_path = os.path.join(folder_path, filename)

            updated_rows = []

            with open(file_path, "r", newline="") as f:
                reader = csv.reader(f)

                for row in reader:

                    if not row:
                        continue

                    # skip header
                    if row[0].lower() == "order":
                        updated_rows.append(row + ["label"])
                        continue

                    # append label
                    updated_rows.append(row + [label_value])

            # overwrite same file
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(updated_rows)


# run it
append_labels_to_csvs("drop_detect")