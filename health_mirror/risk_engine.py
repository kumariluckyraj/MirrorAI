# risk_engine.py
# Hybrid Baseline Risk Engine
# Combines population-level medical grounding (70%)
# with personal deviation detection (30%)

# ── Weights ────────────────────────────────────────────────────
POP_WEIGHT      = 0.7   # population baseline dominates (medical anchor)
PERSONAL_WEIGHT = 0.3   # personal baseline detects individual change

MIN_STD = 0.001         # prevents division by zero


def cap_z(z):
    """Cap z-score at 3 to prevent scores exceeding 100%"""
    return min(max(z, 0), 3)


def z_score_high(value, mean, std):
    """High value = high risk (jaundice, dehydration, stress, skin)"""
    std = max(std, MIN_STD)
    return cap_z((value - mean) / std)


def z_score_low(value, mean, std):
    """Low value = high risk (anemia)"""
    std = max(std, MIN_STD)
    return cap_z((mean - value) / std)


def z_score_deviation(value, mean, std):
    """
    Absolute deviation from personal norm.
    ANY significant change — up or down — from personal baseline is flagged.
    This detects deterioration even when population score is borderline.
    """
    std = max(std, MIN_STD)
    return cap_z(abs(value - mean) / std)


def hybrid_score(z_pop, z_personal, has_personal):
    """
    Combines population and personal z-scores into final risk %.
    
    If no personal baseline exists yet (Day 1), uses population only.
    From Day 2 onwards, blends both with 70/30 weighting.
    
    Returns: float 0.0 to 100.0
    """
    if not has_personal:
        # Day 1 — no personal data yet, full population score
        return round((z_pop / 3) * 100, 2)
    
    combined_z = (POP_WEIGHT * z_pop) + (PERSONAL_WEIGHT * z_personal)
    return round((combined_z / 3) * 100, 2)


# ── Individual Risk Functions ───────────────────────────────────

def anemia_risk(value, pop_baseline, personal_baseline=None):
    """
    Low eyelid redness = anemia risk.
    Population: low A_mean vs population mean
    Personal: significant change from personal norm
    """
    z_pop = z_score_low(
        value,
        pop_baseline["mean"],
        pop_baseline["std"]
    )

    z_personal = 0.0
    has_personal = personal_baseline is not None
    if has_personal:
        z_personal = z_score_deviation(
            value,
            personal_baseline["mean"],
            personal_baseline["std"]
        )

    return hybrid_score(z_pop, z_personal, has_personal)


def jaundice_risk(value, pop_baseline, personal_baseline=None):
    """High sclera yellowness = jaundice risk."""
    z_pop = z_score_high(value, pop_baseline["mean"], pop_baseline["std"])

    z_personal = 0.0
    has_personal = personal_baseline is not None
    if has_personal:
        z_personal = z_score_deviation(
            value,
            personal_baseline["mean"],
            personal_baseline["std"]
        )

    return hybrid_score(z_pop, z_personal, has_personal)


def dehydration_lips_risk(value, pop_baseline, personal_baseline=None):
    """High lip dryness = dehydration risk."""
    z_pop = z_score_high(value, pop_baseline["mean"], pop_baseline["std"])

    z_personal = 0.0
    has_personal = personal_baseline is not None
    if has_personal:
        z_personal = z_score_deviation(
            value,
            personal_baseline["mean"],
            personal_baseline["std"]
        )

    return hybrid_score(z_pop, z_personal, has_personal)


def dehydration_eye_risk(value, pop_baseline, personal_baseline=None):
    """High under-eye darkness = dehydration risk."""
    z_pop = z_score_high(value, pop_baseline["mean"], pop_baseline["std"])

    z_personal = 0.0
    has_personal = personal_baseline is not None
    if has_personal:
        z_personal = z_score_deviation(
            value,
            personal_baseline["mean"],
            personal_baseline["std"]
        )

    return hybrid_score(z_pop, z_personal, has_personal)


def stress_risk(value, pop_baseline, personal_baseline=None):
    """High BPM = stress risk."""
    z_pop = z_score_high(value, pop_baseline["mean"], pop_baseline["std"])

    z_personal = 0.0
    has_personal = personal_baseline is not None
    if has_personal:
        z_personal = z_score_deviation(
            value,
            personal_baseline["mean"],
            personal_baseline["std"]
        )

    return hybrid_score(z_pop, z_personal, has_personal)


def skin_risk(asymmetry, border, color_var, pop_skin, personal_skin=None):
    """
    Three-descriptor mole analysis.
    Averages z-scores across asymmetry, border, and color variance.
    """
    # Population z-scores
    z_asym_pop   = z_score_high(asymmetry,  pop_skin["mole_asymmetry"]["mean"], pop_skin["mole_asymmetry"]["std"])
    z_border_pop = z_score_high(border,     pop_skin["mole_border"]["mean"],    pop_skin["mole_border"]["std"])
    z_color_pop  = z_score_high(color_var,  pop_skin["mole_color_var"]["mean"], pop_skin["mole_color_var"]["std"])
    z_pop = (z_asym_pop + z_border_pop + z_color_pop) / 3

    has_personal = personal_skin is not None
    z_personal = 0.0

    if has_personal:
        z_asym_p   = z_score_deviation(asymmetry, personal_skin["mole_asymmetry"]["mean"], personal_skin["mole_asymmetry"]["std"])
        z_border_p = z_score_deviation(border,     personal_skin["mole_border"]["mean"],    personal_skin["mole_border"]["std"])
        z_color_p  = z_score_deviation(color_var,  personal_skin["mole_color_var"]["mean"], personal_skin["mole_color_var"]["std"])
        z_personal = (z_asym_p + z_border_p + z_color_p) / 3

    return hybrid_score(z_pop, z_personal, has_personal)


# ── Master Function ─────────────────────────────────────────────

def compute_all_risks(features, pop_baseline, personal_baseline=None):
    """
    Master risk computation function.

    Parameters:
        features         — dict of extracted feature values from P2
        pop_baseline     — population baseline (dataset_baseline.json)
        personal_baseline — personal baseline (user_baseline.json) or None on Day 1

    Returns:
        dict of risk percentages for all 6 conditions

    Behaviour:
        - If personal_baseline is None → 100% population weighting (Day 1)
        - If personal_baseline exists  → 70% population + 30% personal deviation
    """
    risks = {}

    # Safe skin baseline extraction
    pop_skin = pop_baseline.get("skin", {
        "mole_asymmetry": pop_baseline.get("mole_asymmetry", {"mean": 0.17, "std": 0.04}),
        "mole_border":    pop_baseline.get("mole_border",    {"mean": 0.20, "std": 0.06}),
        "mole_color_var": pop_baseline.get("mole_color_var", {"mean": 10.4, "std": 2.78})
    })

    personal_skin = None
    if personal_baseline is not None:
        personal_skin = personal_baseline.get("skin", None)

    risks["anemia"] = anemia_risk(
        features["A_mean"],
        pop_baseline["A_mean"],
        personal_baseline["A_mean"] if personal_baseline else None
    )

    risks["jaundice"] = jaundice_risk(
        features["B_mean"],
        pop_baseline["B_mean"],
        personal_baseline["B_mean"] if personal_baseline else None
    )

    risks["dehydration_lips"] = dehydration_lips_risk(
        features["lip_dryness"],
        pop_baseline["lip_dryness"],
        personal_baseline["lip_dryness"] if personal_baseline else None
    )

    risks["dehydration_eye"] = dehydration_eye_risk(
        features["eye_darkness"],
        pop_baseline["eye_darkness"],
        personal_baseline["eye_darkness"] if personal_baseline else None
    )

    risks["stress"] = stress_risk(
        features["BPM"],
        pop_baseline["BPM"],
        personal_baseline["BPM"] if personal_baseline else None
    )

    risks["skin"] = skin_risk(
        features["mole_asymmetry"],
        features["mole_border"],
        features["mole_color_var"],
        pop_skin,
        personal_skin
    )

    return risks