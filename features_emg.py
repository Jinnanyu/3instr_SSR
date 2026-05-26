# -*- coding: utf-8 -*-
"""EMG hand-crafted feature extraction utilities.

Input window shape convention: [T, C]
Output feature vector: concatenated per-channel features [F]

This version keeps your original features and adds several common sEMG
features for cross-subject analysis:
- time/amplitude: rms, iemg, mav, var, wl
- threshold/shape: zc, ssc, wamp, dasdv, logdetector, mavs
- frequency: mnf, mdf

Notes:
1) Threshold-based features use a small adaptive threshold by default.
2) Frequency features use numpy FFT only, so no scipy dependency is required.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

EPS = 1e-8


def _safe_threshold(x: np.ndarray, method: str = "std", factor: float = 0.05) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    if method == "std":
        return float(max(EPS, factor * np.std(x)))
    if method == "absmean":
        return float(max(EPS, factor * np.mean(np.abs(x))))
    return float(max(EPS, factor))


def feature_names_per_channel() -> List[str]:
    return [
        "rms", "iemg", "mav", "var", "wl",
        "zc", "ssc", "wamp", "dasdv", "logdetector", "mavs",
        "mnf", "mdf",
    ]


def _freq_features(x: np.ndarray, sfreq: float) -> Tuple[float, float]:
    """Return mean frequency MNF and median frequency MDF."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size < 4 or sfreq <= 0:
        return 0.0, 0.0
    x = x - float(np.mean(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sfreq))
    power = np.abs(np.fft.rfft(x)) ** 2
    # remove DC component
    if power.size > 1:
        freqs = freqs[1:]
        power = power[1:]
    total_power = float(np.sum(power))
    if total_power <= EPS:
        return 0.0, 0.0
    mnf = float(np.sum(freqs * power) / total_power)
    cumsum = np.cumsum(power)
    mdf_idx = int(np.searchsorted(cumsum, total_power / 2.0))
    mdf_idx = min(max(mdf_idx, 0), len(freqs) - 1)
    mdf = float(freqs[mdf_idx])
    return mnf, mdf


def extract_features_window(
    w_tc: np.ndarray,
    zc_thresh: float | None = None,
    ssc_thresh: float | None = None,
    wamp_thresh: float | None = None,
    sfreq: float = 1000.0,
) -> Tuple[np.ndarray, List[str]]:
    """Extract classic EMG features from one window.

    Parameters
    ----------
    w_tc : ndarray [T, C]
        One EMG window.
    zc_thresh, ssc_thresh, wamp_thresh : float or None
        Optional thresholds. If None, an adaptive per-channel threshold is used.
    sfreq : float
        Sampling rate for frequency-domain features.
    """
    w = np.asarray(w_tc, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"window must be [T,C], got {w.shape}")
    T, C = w.shape
    if T < 3:
        raise ValueError("window too short for feature extraction")

    feats: List[float] = []
    names: List[str] = []

    for c in range(C):
        x = w[:, c].astype(np.float32)
        ax = np.abs(x)
        dx = np.diff(x)

        rms = float(np.sqrt(np.mean(x * x)))
        iemg = float(np.sum(ax))
        mav = float(np.mean(ax))
        var = float(np.var(x))
        wl = float(np.sum(np.abs(dx)))

        zc_thr = _safe_threshold(x) if zc_thresh is None else float(zc_thresh)
        ssc_thr = _safe_threshold(dx) if ssc_thresh is None else float(ssc_thresh)
        wamp_thr = _safe_threshold(x, method="absmean", factor=0.10) if wamp_thresh is None else float(wamp_thresh)

        # Zero crossing with amplitude threshold
        x1 = x[:-1]
        x2 = x[1:]
        zc = int(np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) >= zc_thr)))

        # Slope sign changes with threshold
        d1 = x[1:-1] - x[:-2]
        d2 = x[1:-1] - x[2:]
        ssc = int(np.sum(((d1 * d2) > 0) & ((np.abs(d1) >= ssc_thr) | (np.abs(d2) >= ssc_thr))))

        # Willison amplitude: count adjacent changes above threshold
        wamp = int(np.sum(np.abs(dx) >= wamp_thr))

        # Difference absolute standard deviation value
        dasdv = float(np.sqrt(np.mean(dx * dx))) if dx.size else 0.0

        # Log detector; robustly avoids log(0)
        logdetector = float(np.exp(np.mean(np.log(ax + EPS))))

        # MAV slope: simple first-half vs second-half difference
        mid = max(1, T // 2)
        mavs = float(np.mean(ax[mid:]) - np.mean(ax[:mid])) if mid < T else 0.0

        mnf, mdf = _freq_features(x, sfreq=sfreq)

        vals = [
            rms, iemg, mav, var, wl,
            float(zc), float(ssc), float(wamp), dasdv, logdetector, mavs,
            mnf, mdf,
        ]
        feats.extend(vals)
        names.extend([f"ch{c+1}_{n}" for n in feature_names_per_channel()])

    return np.asarray(feats, dtype=np.float32), names


def extract_features_batch(X_tcn: np.ndarray, sfreq: float = 1000.0) -> Tuple[np.ndarray, List[str]]:
    X = np.asarray(X_tcn, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must be [N,T,C], got {X.shape}")
    out = []
    names: List[str] | None = None
    for i in range(X.shape[0]):
        feat, cur_names = extract_features_window(X[i], sfreq=sfreq)
        if names is None:
            names = cur_names
        out.append(feat)
    return np.stack(out, axis=0).astype(np.float32), (names or [])
