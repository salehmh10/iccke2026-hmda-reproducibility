"""Pure literature-method helpers for Prompt 4B4.

The functions in this module have no project file access. Target-derived
quantities are training-loss inputs only and are never inference features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.special import logsumexp


SEED = 42
SERA_T = 1000
PROHIBITED_INFERENCE_FIELDS = {
    "respondent_id", "p_tail", "global_prediction", "residual_proposal",
    "component_disagreement", "benefit_score", "beat_probability",
    "target_rarity_score", "target_decile", "true_tail", "realized_error",
    "residual", "benefit",
}


def finite_vector(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and non-empty.")
    return result


def validate_feature_contract(features: Sequence[str]) -> list[str]:
    names = list(features)
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError("Prompt 4B4 requires exactly 35 unique ordered features.")
    bad = sorted(set(names) & PROHIBITED_INFERENCE_FIELDS)
    if bad:
        raise ValueError(f"Prohibited inference features: {bad}")
    return names


def denseweight_official(y: Any, alpha: float = 1.0, grid_points: int = 4096):
    """Run the official DenseWeight 0.1.2 implementation."""
    if alpha != 1.0 or grid_points != 4096:
        raise ValueError("Prompt 4B4 freezes DenseWeight alpha=1 and grid_points=4096.")
    from denseweight import DenseWeight

    target = finite_vector(y, "y")
    estimator = DenseWeight(alpha=alpha)
    weights = np.asarray(estimator.fit(target, grid_points=grid_points), dtype=np.float64)
    if weights.shape != target.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("Official DenseWeight returned invalid weights.")
    return weights, estimator


def denseweight_direct_from_density(density: Any, alpha: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    density = finite_vector(density, "density")
    if np.any((density < -1e-12) | (density > 1 + 1e-12)):
        raise ValueError("DenseWeight normalized density must be in [0, 1].")
    density = np.clip(density, 0.0, 1.0)
    raw = np.maximum(1.0 - alpha * density, eps)
    return raw / raw.mean()


def sera_sigma(relevance: Any, t: int = SERA_T) -> np.ndarray:
    """Exact authors' trapezoidal coefficient sigma(phi) for T intervals."""
    phi = finite_vector(relevance, "relevance")
    if t != 1000 or np.any((phi < 0.0) | (phi > 1.0)):
        raise ValueError("Prompt 4B4 freezes T=1000 and relevance in [0,1].")
    interior = np.arange(1, t, dtype=np.float64) / t
    count = np.searchsorted(interior, phi, side="right").astype(np.float64)
    coefficient = (0.5 + count + 0.5 * np.isclose(phi, 1.0, atol=0.0, rtol=0.0)) / t
    return coefficient


def sera_bruteforce_sigma(relevance: Any, t: int = SERA_T) -> np.ndarray:
    phi = finite_vector(relevance, "relevance")
    steps = np.linspace(0.0, 1.0, t + 1)
    result = []
    for value in phi:
        indicators = (value >= steps).astype(np.float64)
        result.append(float(np.trapezoid(indicators, steps)))
    return np.asarray(result)


@dataclass
class SERAObjective:
    coefficient: np.ndarray

    def __post_init__(self):
        self.coefficient = finite_vector(self.coefficient, "coefficient")
        if np.any(self.coefficient <= 0):
            raise ValueError("SERA coefficients must be positive.")

    def __call__(self, y_true: Any, y_pred: Any):
        actual = finite_vector(y_true, "y_true")
        predicted = finite_vector(y_pred, "y_pred")
        if actual.shape != predicted.shape or actual.shape != self.coefficient.shape:
            raise ValueError("SERA objective vectors are not aligned.")
        gradient = 2.0 * self.coefficient * (predicted - actual)
        hessian = 2.0 * self.coefficient
        return gradient, hessian


def lds_kernel_window(kernel: str = "gaussian", ks: int = 5, sigma: float = 2.0) -> np.ndarray:
    if (kernel, ks, sigma) != ("gaussian", 5, 2.0):
        raise ValueError("Prompt 4B4 freezes Gaussian LDS ks=5 sigma=2.")
    half = (ks - 1) // 2
    base = np.asarray([0.0] * half + [1.0] + [0.0] * half)
    window = gaussian_filter1d(base, sigma=sigma)
    return window / window.max()


def lds_weights(y: Any, bin_width: float = 1.0):
    """Official sqrt-inverse LDS adaptation with fixed natural-unit bins."""
    target = finite_vector(y, "y")
    if bin_width != 1.0 or np.any(target < 0):
        raise ValueError("Prompt 4B4 freezes non-negative unit-width target bins.")
    bins = np.floor(target / bin_width).astype(np.int64)
    counts = np.bincount(bins, minlength=int(bins.max()) + 1).astype(np.float64)
    sqrt_frequency = np.sqrt(counts)
    effective = convolve1d(sqrt_frequency, weights=lds_kernel_window(), mode="constant")
    row_effective = effective[bins]
    if np.any(row_effective <= 0):
        raise RuntimeError("Occupied LDS bins must have positive effective density.")
    weights = 1.0 / row_effective
    weights *= len(weights) / weights.sum()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("LDS returned invalid weights.")
    return weights, bins, counts, effective


def fit_imr_gmm(y: Any, n_components: int = 6, random_state: int = SEED):
    """Fit the fixed standard IMr-GB six-component Train-label prior."""
    if n_components != 6 or random_state != 42:
        raise ValueError("Prompt 4B4 freezes standard IMr-GB to six GMM components and seed 42.")
    from sklearn.mixture import GaussianMixture

    target = finite_vector(y, "y")
    model = GaussianMixture(n_components=6, random_state=42, covariance_type="full")
    model.fit(target.reshape(-1, 1))
    means = model.means_.reshape(-1).astype(np.float64)
    weights = model.weights_.reshape(-1).astype(np.float64)
    variances = model.covariances_.reshape(-1).astype(np.float64)
    if not (np.isfinite(means).all() and np.isfinite(weights).all() and np.isfinite(variances).all()):
        raise RuntimeError("IMr prior contains non-finite parameters.")
    if np.any(weights <= 0) or np.any(variances <= 0) or not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("IMr prior parameters are invalid.")
    return model, means, weights, variances


def imr_gradient(y_true: Any, y_pred: Any, means: Any, weights: Any, variances: Any, noise_var: float = 1.0):
    """Tutorial-consistent Balanced-MSE gradient used by IMr-GB.

    The authors' implementation deliberately supplies a constant unit Hessian.
    """
    actual = finite_vector(y_true, "y_true")
    predicted = finite_vector(y_pred, "y_pred")
    mu = finite_vector(means, "means")
    pi = finite_vector(weights, "weights")
    var = finite_vector(variances, "variances")
    if actual.shape != predicted.shape or not (mu.shape == pi.shape == var.shape):
        raise ValueError("IMr vectors are not aligned.")
    total_var = var + float(noise_var)
    diff = predicted[:, None] - mu[None, :]
    log_component = (
        np.log(pi)[None, :] - 0.5 * np.log(2.0 * np.pi * total_var)[None, :]
        - 0.5 * diff**2 / total_var[None, :]
    )
    responsibility = np.exp(log_component - logsumexp(log_component, axis=1, keepdims=True))
    prior_score = np.sum(responsibility * (-diff / total_var[None, :]), axis=1)
    gradient = 2.0 * (predicted - actual) + 2.0 * float(noise_var) * prior_score
    return gradient, np.ones_like(gradient)


@dataclass
class IMrObjective:
    means: np.ndarray
    weights: np.ndarray
    variances: np.ndarray
    noise_var: float = 1.0

    def __post_init__(self):
        self.means = finite_vector(self.means, "means")
        self.weights = finite_vector(self.weights, "weights")
        self.variances = finite_vector(self.variances, "variances")

    def __call__(self, y_true: Any, y_pred: Any):
        return imr_gradient(y_true, y_pred, self.means, self.weights, self.variances, self.noise_var)


def six_condition_evaluator(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "C1_overall_mae_improves": float(candidate["mae"]) < float(reference["mae"]),
        "C2_top_decile_mae_improves_3pct": float(candidate["top_decile_mae"]) <= float(reference["top_decile_mae"]) * 0.97,
        "C3_bottom_90_mae_worsens_at_most_0_25pct": float(candidate["bottom_90_mae"]) <= float(reference["bottom_90_mae"]) * 1.0025,
        "C4_rmse_worsens_at_most_0_25pct": float(candidate["rmse"]) <= float(reference["rmse"]) * 1.0025,
        "C5_top_decile_signed_error_closer_to_zero": abs(float(candidate["top_decile_signed_error"])) < abs(float(reference["top_decile_signed_error"])),
        "C6_top_decile_underprediction_rate_decreases": float(candidate["top_decile_underprediction_rate"]) < float(reference["top_decile_underprediction_rate"]),
    }
    passed = int(sum(checks.values()))
    return {**checks, "conditions_passed": passed, "conditions_total": 6,
            "status": "PASS" if passed == 6 else "PARTIAL" if passed else "FAIL"}


def select_tailaware(rows: Sequence[Mapping[str, Any]]) -> str:
    return min(rows, key=lambda r: (-int(r["conditions_passed"]), float(r["mae"]),
                                    float(r["top_decile_mae"]), float(r["rmse"]),
                                    int(r.get("complexity", 99)), str(r["candidate_id"]))) ["candidate_id"]


def select_mae_challenger(rows: Sequence[Mapping[str, Any]]) -> str:
    return min(rows, key=lambda r: (float(r["mae"]), float(r["rmse"]),
                                    float(r["top_decile_mae"]), int(r.get("complexity", 99)),
                                    str(r["candidate_id"]))) ["candidate_id"]


def fixed_substitution_predictions(densecat: Any, seraxgb: Any, imrgb: Any, ldslgb: Any,
                                   frozen_cat: Any, frozen_lgb: Any, frozen_xgb: Any) -> dict[str, np.ndarray]:
    """Apply only the four predeclared 0.60/0.20/0.20 substitutions."""
    vectors = [finite_vector(v, "prediction") for v in
               (densecat, seraxgb, imrgb, ldslgb, frozen_cat, frozen_lgb, frozen_xgb)]
    if len({v.shape for v in vectors}) != 1:
        raise ValueError("Substitution prediction vectors are not aligned.")
    densecat, seraxgb, imrgb, ldslgb, frozen_cat, frozen_lgb, frozen_xgb = vectors
    return {
        "prompt4b4__sub_densecat": .60*densecat + .20*frozen_lgb + .20*frozen_xgb,
        "prompt4b4__sub_seraxgb": .60*frozen_cat + .20*frozen_lgb + .20*seraxgb,
        "prompt4b4__sub_imrgb": .60*frozen_cat + .20*frozen_lgb + .20*imrgb,
        "prompt4b4__sub_ldslgb": .60*frozen_cat + .20*ldslgb + .20*frozen_xgb,
    }


def scientific_resume_action(entries: Sequence[Mapping[str, Any]], candidate_id: str,
                             bundle_exists: bool, prediction_exists: bool) -> str:
    """Return the bounded idempotent action for one registered Candidate."""
    attempts = [row for row in entries if row.get("category") == "scientific_candidate"
                and row.get("candidate_id") == candidate_id]
    passed = [row for row in attempts if row.get("status") == "PASS"]
    if passed:
        if not (bundle_exists and prediction_exists):
            raise RuntimeError("A PASS fit is missing a required persisted artifact.")
        return "REUSE"
    if len(attempts) >= 2:
        raise RuntimeError("The identical technical retry budget is exhausted.")
    return "START"
