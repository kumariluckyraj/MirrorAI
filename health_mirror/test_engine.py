import json
from risk_engine import compute_all_risks

# -------------------------
# Step 1: Load baseline
# -------------------------
with open("dataset_baseline.json", "r") as f:
    baseline = json.load(f)

# -------------------------
# Step 2: Fake extracted features
# (Simulating real module outputs)
# -------------------------
features = {
    "A_mean": 140,              # lower → anemia risk
    "B_mean": 138,              # higher → jaundice risk
    "lip_dryness": 0.5,         # high → dehydration
    "eye_darkness": 0.4,        # high → dehydration
    "BPM": 98,                  # high → stress
    "mole_asymmetry": 0.35,     # high → skin risk
    "mole_border": 0.4,
    "mole_color_var": 18
}

# -------------------------
# Step 3: Run Engine
# -------------------------
risks = compute_all_risks(features, baseline)

# -------------------------
# Step 4: Print Results
# -------------------------
print("Module Risks:")
for module, value in risks.items():
    print(f"{module}: {value}%")

