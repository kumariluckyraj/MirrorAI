import random
import csv
import json
import pandas as pd

rows = []

for i in range(200):
    sample = [
        random.uniform(145,155),     # A_mean
        random.uniform(128,138),     # B_mean
        random.uniform(0.2,0.4),     # lip dryness
        random.uniform(0.2,0.35),    # eye darkness
        random.uniform(65,85),       # BPM
        random.uniform(0.1,0.25),    # mole asymmetry
        random.uniform(0.1,0.3),     # mole border
        random.uniform(5,15)         # mole color var
    ]
    rows.append(sample)

with open("synthetic_dataset.csv","w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "A_mean","B_mean","lip_dryness",
        "eye_darkness","BPM",
        "mole_asymmetry","mole_border","mole_color_var"
    ])
    writer.writerows(rows)

print("Dataset Created!")


df = pd.read_csv("synthetic_dataset.csv")
baseline = {}

for column in df.columns:
    baseline[column] = {
        "mean": df[column].mean(),
        "std": df[column].std()
    }
print("Dataset baseline Created")

with open("dataset_baseline.json","w") as f:
    json.dump(baseline,f,indent=4)

