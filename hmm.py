import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler


# -----------------------------
# CONFIG
# -----------------------------
N_STATES     = 3
RANDOM_STATE = 42
N_ITER       = 1000
VOL_WINDOW   = 20


# -----------------------------
# FEATURE BUILDER
# -----------------------------
def build_features(df, returns, vol_window=VOL_WINDOW):
    """
    Constructs a (T, 3) feature matrix:
        - log returns
        - realised volatility (rolling std of returns)
        - log volume change
    """
    realised_vol = np.array([
        np.std(returns[max(0, i - vol_window):i + 1], ddof=0)
        for i in range(len(returns))
    ])

    volume      = df["volume"].values
    log_vol_chg = np.diff(np.log(volume + 1))

    min_len  = min(len(returns), len(realised_vol), len(log_vol_chg))
    features = np.column_stack([
        returns[-min_len:],
        realised_vol[-min_len:],
        log_vol_chg[-min_len:]
    ])

    mask     = np.isfinite(features).all(axis=1)
    features = features[mask]

    return features


# -----------------------------
# FIT HMM
# -----------------------------
def fit_hmm(features, n_states=N_STATES):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        tol=1e-4
    )
    model.fit(features)
    return model


# -----------------------------
# LABEL STATES
# -----------------------------
def label_states(model, scaler, n_states=N_STATES):
    means_scaled = model.means_
    means_real   = scaler.inverse_transform(means_scaled)

    mean_returns = means_real[:, 0]
    mean_vol     = means_real[:, 1]

    sorted_by_return = np.argsort(mean_returns)

    bear_state  = sorted_by_return[0]
    bull_state  = sorted_by_return[-1]
    remaining   = [s for s in range(n_states) if s not in [bear_state, bull_state]]
    panic_state = max(remaining, key=lambda s: mean_vol[s]) if remaining else sorted_by_return[1]

    return {
        bear_state:  "high_vol",
        bull_state:  "low_vol",
        panic_state: "extreme_vol"
    }


# -----------------------------
# DECODE REGIMES
# -----------------------------
def decode_regimes(model, features, label_map):
    hidden_states = model.predict(features)
    return np.array([label_map[s] for s in hidden_states])


# -----------------------------
# TRANSITION MATRIX
# -----------------------------
def get_transition_matrix(model, label_map, n_states=N_STATES):
    trans  = model.transmat_
    labels = [label_map[i] for i in range(n_states)]

    matrix = {}
    for i, from_label in enumerate(labels):
        matrix[from_label] = {}
        for j, to_label in enumerate(labels):
            matrix[from_label][to_label] = trans[i][j]

    return matrix


# -----------------------------
# STATE MEANS (for interpretability)
# -----------------------------
def get_state_means(model, scaler, label_map):
    """
    Returns the real-space mean return and vol for each labelled state.
    Tells you whether 'bull' actually means positive returns or just least bad.
    """
    means_real = scaler.inverse_transform(model.means_)

    result = {}
    for state_int, label in label_map.items():
        result[label] = {
            "mean_return": means_real[state_int, 0],
            "mean_vol":    means_real[state_int, 1],
        }

    return result


# -----------------------------
# DEPENDABILITY SCORE
# -----------------------------
def compute_dependability(model, features, regimes, label_map, state_means):
    """
    Scores how much you can trust this HMM fit on 5 dimensions.
    Each dimension scored 0-20, total out of 100.

    Dimensions:
        1. State separation    — are the regime means actually different?
        2. Regime stability    — does the model stay in states (not flip every day)?
        3. Convergence         — did the EM algorithm actually converge?
        4. Data sufficiency    — enough data to fit reliably?
        5. Label validity      — does 'bull' actually have positive mean return?
    """
    scores  = {}
    details = {}

    # --- 1. STATE SEPARATION (0-20) ---
    # measure how far apart the state mean returns are relative to their spread
    # similar to a between-class vs within-class ratio
    mean_returns = np.array([state_means[label_map[i]]["mean_return"] for i in range(N_STATES)])
    mean_vols    = np.array([state_means[label_map[i]]["mean_vol"]    for i in range(N_STATES)])

    return_spread = np.max(mean_returns) - np.min(mean_returns)
    avg_vol       = np.mean(mean_vols)

    # separation ratio: how wide the spread is relative to average within-state noise
    sep_ratio = return_spread / (avg_vol + 1e-8)

    if sep_ratio > 1.5:
        sep_score = 20
    elif sep_ratio > 1.0:
        sep_score = 15
    elif sep_ratio > 0.5:
        sep_score = 10
    elif sep_ratio > 0.2:
        sep_score = 5
    else:
        sep_score = 0

    scores["state_separation"]  = sep_score
    details["state_separation"] = f"Return spread / avg vol = {sep_ratio:.3f}"

    # --- 2. REGIME STABILITY (0-20) ---
    # average self-transition probability across all states
    # high = model stays in a state once it enters (good)
    # low  = model flips constantly (unreliable)
    self_trans = np.mean([model.transmat_[i, i] for i in range(N_STATES)])

    if self_trans > 0.90:
        stab_score = 20
    elif self_trans > 0.80:
        stab_score = 15
    elif self_trans > 0.70:
        stab_score = 10
    elif self_trans > 0.60:
        stab_score = 5
    else:
        stab_score = 0

    scores["regime_stability"]  = stab_score
    details["regime_stability"] = f"Avg self-transition prob = {self_trans:.3f}"

    # --- 3. CONVERGENCE (0-20) ---
    # hmmlearn exposes monitor_.converged
    converged = model.monitor_.converged

    if converged:
        conv_score = 20
        conv_msg   = "EM converged within iteration limit"
    else:
        conv_score = 5
        conv_msg   = "EM did NOT converge — fit may be unstable"

    scores["convergence"]  = conv_score
    details["convergence"] = conv_msg

    # --- 4. DATA SUFFICIENCY (0-20) ---
    # rule of thumb: need at least 100 obs per state for stable Gaussian estimation
    n_obs = len(features)
    obs_per_state = n_obs / N_STATES

    if obs_per_state > 200:
        data_score = 20
    elif obs_per_state > 100:
        data_score = 15
    elif obs_per_state > 50:
        data_score = 8
    else:
        data_score = 2

    scores["data_sufficiency"]  = data_score
    details["data_sufficiency"] = f"{n_obs} total obs ({obs_per_state:.0f} per state avg)"

    # --- 5. LABEL VALIDITY (0-20) ---
    # checks whether the labels actually make directional sense
    # bull should have higher mean return than bear
    # panic should have higher vol than bull and bear
    bull_return  = state_means["low_vol"]["mean_return"]
    bear_return  = state_means["high_vol"]["mean_return"]
    panic_vol    = state_means["extreme_vol"]["mean_vol"]
    bull_vol     = state_means["low_vol"]["mean_vol"]
    bear_vol     = state_means["high_vol"]["mean_vol"]

    direction_ok = bull_return > bear_return        # bull beats bear on return
    panic_vol_ok = panic_vol > max(bull_vol, bear_vol)  # panic is most volatile

    # whether bull is actually positive (vs just least bad)
    bull_positive = bull_return > 0

    if direction_ok and panic_vol_ok and bull_positive:
        label_score = 20
        label_msg   = "All labels valid — bull positive, panic most volatile"
    elif direction_ok and panic_vol_ok:
        label_score = 12
        label_msg   = "Structure valid but bull mean return is negative (relative labelling only)"
    elif direction_ok:
        label_score = 8
        label_msg   = "Return ordering correct but panic not most volatile"
    else:
        label_score = 0
        label_msg   = "Labels may be unreliable — states not well separated by direction"

    scores["label_validity"]  = label_score
    details["label_validity"] = label_msg

    # --- TOTAL ---
    total = sum(scores.values())

    if total >= 80:
        verdict = "HIGH   — trust the regime signals"
    elif total >= 60:
        verdict = "MEDIUM — use with caution, cross-reference momentum"
    elif total >= 40:
        verdict = "LOW    — treat as directional hint only"
    else:
        verdict = "POOR   — do not trade on this signal"

    return {
        "total":   total,
        "verdict": verdict,
        "scores":  scores,
        "details": details
    }


# -----------------------------
# REGIME SUMMARY
# -----------------------------
def regime_summary(regimes):
    total = len(regimes)
    for label in ["low_vol", "high_vol", "extreme_vol"]:
        count = np.sum(regimes == label)
        print(f"  {label.capitalize():6s}: {count:4d} days  ({count/total*100:.1f}%)")

    print(f"\n  Current regime: {regimes[-1].upper()}")


# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
def run_hmm(asset):
    features_raw = build_features(asset.df, asset.returns)

    if len(features_raw) < 50:
        raise ValueError("Not enough clean rows to fit HMM after NaN removal.")

    scaler          = RobustScaler()
    features_scaled = scaler.fit_transform(features_raw)

    model       = fit_hmm(features_scaled)
    label_map   = label_states(model, scaler)
    regimes     = decode_regimes(model, features_scaled, label_map)
    state_means = get_state_means(model, scaler, label_map)
    dep_score   = compute_dependability(model, features_scaled, regimes, label_map, state_means)
    trans_matrix = get_transition_matrix(model, label_map)

    asset.regimes      = regimes
    asset.hmm_model    = model
    asset.hmm_features = features_scaled

    return label_map, trans_matrix, state_means, dep_score