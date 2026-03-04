# risk_engine.py

def cap_z(z):
    """Cap z-score at 3 to avoid extreme inflation"""
    return min(z, 3)


# -------------------------
# ANEMIA (low redness risk)
# -------------------------
def anemia_risk(value, mean, std):
    # lower than mean is risky
    z = (mean - value) / std

    if z <= 0:
        return 0.0

    z = cap_z(z)
    return round((z / 3) * 100, 2)


# -------------------------
# JAUNDICE (high yellow risk)
# -------------------------
def jaundice_risk(value, mean, std):
    z = (value - mean) / std

    if z <= 0:
        return 0.0

    z = cap_z(z)
    return round((z / 3) * 100, 2)


# -------------------------
# DEHYDRATION (lips dryness)
# -------------------------
def dehydration_lips_risk(value, mean, std):
    z = (value - mean) / std

    if z <= 0:
        return 0.0

    z = cap_z(z)
    return round((z / 3) * 100, 2)


# -------------------------
# DEHYDRATION (under-eye)
# -------------------------
def dehydration_eye_risk(value, mean, std):
    z = (value - mean) / std

    if z <= 0:
        return 0.0

    z = cap_z(z)
    return round((z / 3) * 100, 2)


# -------------------------
# STRESS (BPM)
# -------------------------
def stress_risk(bpm, mean, std):
    z = (bpm - mean) / std

    if z <= 0:
        return 0.0

    z = cap_z(z)
    return round((z / 3) * 100, 2)


# -------------------------
# SKIN / MOLE
# -------------------------
def skin_risk(asymmetry, border, color_var, baseline_skin):

    z_asym = (asymmetry - baseline_skin["mole_asymmetry"]["mean"]) / baseline_skin["mole_asymmetry"]["std"]
    z_border = (border - baseline_skin["mole_border"]["mean"]) / baseline_skin["mole_border"]["std"]
    z_color = (color_var - baseline_skin["mole_color_var"]["mean"]) / baseline_skin["mole_color_var"]["std"]

    # Only high deviation matters
    z_asym = min(max(0, z_asym), 3)
    z_border = min(max(0, z_border), 3)
    z_color = min(max(0, z_color), 3)

    avg_z = (z_asym + z_border + z_color) / 3

    return round((avg_z / 3) * 100, 2)
# -------------------------
# MASTER ENGINE
# -------------------------
def compute_all_risks(features, baseline):
    risks = {}

    risks["anemia"] = anemia_risk(
        features["A_mean"],
        baseline["A_mean"]["mean"],
        baseline["A_mean"]["std"]
    )

    risks["jaundice"] = jaundice_risk(
        features["B_mean"],
        baseline["B_mean"]["mean"],
        baseline["B_mean"]["std"]
    )

    risks["dehydration_lips"] = dehydration_lips_risk(
        features["lip_dryness"],
        baseline["lip_dryness"]["mean"],
        baseline["lip_dryness"]["std"]
    )

    risks["dehydration_eye"] = dehydration_eye_risk(
        features["eye_darkness"],
        baseline["eye_darkness"]["mean"],
        baseline["eye_darkness"]["std"]
    )

    risks["stress"] = stress_risk(
        features["BPM"],
        baseline["BPM"]["mean"],
        baseline["BPM"]["std"]
    )

    risks["skin"] = skin_risk(
    features["mole_asymmetry"],
    features["mole_border"],
    features["mole_color_var"],
    baseline["skin"]   
)

    return risks

