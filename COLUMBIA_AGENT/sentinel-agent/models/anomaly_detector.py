"""
SENTINEL — ML Anomaly Detection Pipeline

Models:
  1. Z-Score streaming detector (fast)
  2. Isolation Forest (batch, high accuracy)
  3. Volatility regime change detector
  4. Order flow imbalance detector
  5. Ensemble scorer combining all models
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from utils import get_logger

log = get_logger("ml")


def _sanitize(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ─── Rolling Statistics (Online) ──────────────────────

class RollingStats:
    """Efficient online mean/std computation."""

    def __init__(self, window: int = 200):
        self.window = window
        self.values: deque = deque(maxlen=window)
        self._sum = 0.0
        self._sum_sq = 0.0

    def push(self, val: float):
        if len(self.values) == self.window:
            old = self.values[0]
            self._sum -= old
            self._sum_sq -= old ** 2
        self.values.append(val)
        self._sum += val
        self._sum_sq += val ** 2

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return self._sum / self.n if self.n else 0.0

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1e-10
        var = (self._sum_sq / self.n) - (self._sum / self.n) ** 2
        return max(var, 0) ** 0.5

    @property
    def zscore(self) -> float:
        s = self.std
        if s < 1e-10 or not self.values:
            return 0.0
        return (self.values[-1] - self.mean) / s

    @property
    def last(self) -> float:
        return self.values[-1] if self.values else 0.0


# ─── Feature Extractor ───────────────────────────────

@dataclass
class FeatureVector:
    timestamp: float = 0.0
    price_return: float = 0.0
    log_return: float = 0.0
    volume_ratio: float = 1.0
    volatility_ratio: float = 1.0
    spread_bps: float = 0.0
    order_imbalance: float = 0.0
    price_acceleration: float = 0.0
    tick_intensity: float = 0.0


class FeatureExtractor:
    """Converts raw market ticks into feature vectors."""

    def __init__(self):
        self.prev_price = None
        self.prev_return = 0.0
        self.price_stats = RollingStats(100)
        self.volume_stats = RollingStats(100)
        self.vol_stats = RollingStats(50)
        self.tick_times: deque = deque(maxlen=100)

    def extract(self, price: float, volume: int, timestamp: float) -> FeatureVector:
        # Returns
        ret = 0.0
        log_ret = 0.0
        if self.prev_price and self.prev_price > 0:
            ret = (price - self.prev_price) / self.prev_price
            log_ret = np.log(price / self.prev_price) if price > 0 else 0.0

        # Acceleration
        accel = ret - self.prev_return

        # Volume ratio
        self.volume_stats.push(float(volume))
        vol_ratio = volume / max(self.volume_stats.mean, 1.0)

        # Volatility
        self.vol_stats.push(abs(log_ret))
        short_vol = np.mean(list(self.vol_stats.values)[-10:]) if self.vol_stats.n >= 10 else self.vol_stats.mean
        long_vol = self.vol_stats.mean
        vol_ratio_val = short_vol / max(long_vol, 1e-10)

        # Tick intensity
        self.tick_times.append(timestamp)
        intensity = 0.0
        if len(self.tick_times) > 1:
            dt = self.tick_times[-1] - self.tick_times[0]
            if dt > 0:
                intensity = len(self.tick_times) / dt

        # Order imbalance (simulated from price movement + volume)
        imbalance = np.tanh(ret * 100) * min(vol_ratio, 3.0) / 3.0

        self.prev_price = price
        self.prev_return = ret

        return FeatureVector(
            timestamp=timestamp,
            price_return=ret,
            log_return=log_ret,
            volume_ratio=vol_ratio,
            volatility_ratio=vol_ratio_val,
            order_imbalance=imbalance,
            price_acceleration=accel,
            tick_intensity=intensity,
        )


# ─── Detection Result ────────────────────────────────

@dataclass
class DetectionResult:
    model_name: str
    score: float              # 0.0 = normal, 1.0 = extreme anomaly
    is_anomaly: bool
    anomaly_type: str = "NONE"
    details: Dict = field(default_factory=dict)


# ─── Model 1: Z-Score Detector ───────────────────────

class ZScoreDetector:
    """Streaming z-score detector across multiple dimensions."""

    def __init__(self, threshold: float = 2.5, window: int = 200):
        self.threshold = threshold
        self.return_stats = RollingStats(window)
        self.volume_stats = RollingStats(window)
        self.vol_stats = RollingStats(window)

    def score(self, fv: FeatureVector) -> DetectionResult:
        self.return_stats.push(fv.log_return)
        self.volume_stats.push(fv.volume_ratio)
        self.vol_stats.push(fv.volatility_ratio)

        z_ret = abs(self.return_stats.zscore)
        z_vol = abs(self.volume_stats.zscore)
        z_vr = abs(self.vol_stats.zscore)

        composite = z_ret * 0.35 + z_vol * 0.40 + z_vr * 0.25
        normalized = min(1.0, composite / (self.threshold * 2))

        atype = "NONE"
        if composite > self.threshold:
            if z_vol > z_ret and z_vol > z_vr:
                atype = "VOLUME_SPIKE"
            elif z_ret > self.threshold * 1.2:
                atype = "PRICE_DISLOCATION"
            else:
                atype = "STATISTICAL_OUTLIER"

        return DetectionResult(
            model_name="zscore",
            score=round(normalized, 4),
            is_anomaly=composite > self.threshold,
            anomaly_type=atype,
            details={"z_return": round(z_ret, 3), "z_volume": round(z_vol, 3), "z_volatility": round(z_vr, 3)},
        )


# ─── Model 2: Isolation Forest (Lite) ────────────────

class IsolationForestLite:
    """Lightweight streaming isolation forest."""

    def __init__(self, n_trees: int = 50, sample_size: int = 128, max_depth: int = 10):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.buffer: deque = deque(maxlen=512)
        self.trees: list = []
        self._fitted = False
        self._counter = 0

    def _build_tree(self, data: np.ndarray, depth: int = 0) -> dict:
        n, f = data.shape
        if depth >= self.max_depth or n <= 2:
            return {"leaf": True, "size": n}
        fi = np.random.randint(0, f)
        col = data[:, fi]
        lo, hi = col.min(), col.max()
        if abs(hi - lo) < 1e-10:
            return {"leaf": True, "size": n}
        split = np.random.uniform(lo, hi)
        left = col < split
        return {
            "leaf": False, "feature": fi, "split": split,
            "left": self._build_tree(data[left], depth + 1),
            "right": self._build_tree(data[~left], depth + 1),
        }

    def _path_len(self, x: np.ndarray, tree: dict, depth: int = 0) -> float:
        if tree["leaf"]:
            n = tree["size"]
            if n <= 1:
                return float(depth)
            return depth + 2.0 * (np.log(max(n - 1, 1)) + 0.5772) - 2.0 * (n - 1) / max(n, 1)
        if x[tree["feature"]] < tree["split"]:
            return self._path_len(x, tree["left"], depth + 1)
        return self._path_len(x, tree["right"], depth + 1)

    def _fit(self):
        if len(self.buffer) < self.sample_size:
            return
        data = np.array(list(self.buffer))
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(len(data), min(self.sample_size, len(data)), replace=False)
            self.trees.append(self._build_tree(data[idx]))
        self._fitted = True

    def score(self, fv: FeatureVector) -> DetectionResult:
        vec = np.array([fv.log_return, fv.volume_ratio, fv.volatility_ratio,
                        fv.order_imbalance, fv.price_acceleration, fv.tick_intensity])
        self.buffer.append(vec)
        self._counter += 1
        if self._counter % 64 == 0:
            self._fit()

        if not self._fitted:
            return DetectionResult("iforest", 0.0, False, details={"status": "warming_up"})

        avg_path = np.mean([self._path_len(vec, t) for t in self.trees])
        n = self.sample_size
        c_n = 2.0 * (np.log(max(n - 1, 1)) + 0.5772) - 2.0 * (n - 1) / max(n, 1)
        sc = float(2 ** (-avg_path / max(c_n, 1e-10)))

        return DetectionResult("iforest", round(sc, 4), sc > 0.62,
                               "STATISTICAL_OUTLIER" if sc > 0.62 else "NONE",
                               {"avg_path": round(avg_path, 2)})


# ─── Model 3: Volatility Regime Detector ─────────────

class VolatilityRegimeDetector:
    def __init__(self):
        self.short_vol = RollingStats(20)
        self.long_vol = RollingStats(100)

    def score(self, fv: FeatureVector) -> DetectionResult:
        self.short_vol.push(abs(fv.log_return))
        self.long_vol.push(abs(fv.log_return))

        if self.long_vol.n < 50:
            return DetectionResult("vol_regime", 0.0, False, details={"status": "warming_up"})

        ratio = self.short_vol.mean / max(self.long_vol.mean, 1e-10)
        is_anom = ratio > 2.5 or ratio < 0.2
        sc = min(1.0, abs(ratio - 1.0) / 3.0) if is_anom else 0.0

        regime = "NORMAL"
        if ratio > 2.5:
            regime = "HIGH_VOL"
        elif ratio < 0.2:
            regime = "SUPPRESSED"

        return DetectionResult("vol_regime", round(sc, 4), is_anom,
                               "VOLATILITY_REGIME" if is_anom else "NONE",
                               {"ratio": round(ratio, 3), "regime": regime})


# ─── Model 4: Order Flow Detector ────────────────────

class OrderFlowDetector:
    def __init__(self):
        self.imbalance_stats = RollingStats(50)

    def score(self, fv: FeatureVector) -> DetectionResult:
        self.imbalance_stats.push(fv.order_imbalance)

        z = abs(self.imbalance_stats.zscore)

        # Wash trade detection: rapid sign flips
        wash_score = 0.0
        if self.imbalance_stats.n >= 10:
            recent = list(self.imbalance_stats.values)[-10:]
            flips = sum(1 for i in range(1, len(recent)) if recent[i] * recent[i - 1] < 0)
            wash_score = flips / 9.0

        spoof_score = min(1.0, z / 4.0)
        composite = max(spoof_score, wash_score)

        atype = "NONE"
        if composite > 0.55:
            if wash_score > spoof_score:
                atype = "WASH_TRADE"
            elif z > 3.0:
                atype = "SPOOFING"
            else:
                atype = "LAYERING"

        return DetectionResult("order_flow", round(composite, 4), composite > 0.55,
                               atype, {"z_imbalance": round(z, 3), "wash_score": round(wash_score, 3)})


# ═══════════════════════════════════════════════════════
# ENSEMBLE DETECTOR
# ═══════════════════════════════════════════════════════

class AnomalyDetectorML:
    """
    Ensemble anomaly detector combining all models.
    Each tick → feature extraction → all models score → weighted ensemble.
    """

    WEIGHTS = {
        "zscore": 0.30,
        "iforest": 0.30,
        "vol_regime": 0.20,
        "order_flow": 0.20,
    }

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.zscore = ZScoreDetector()
        self.iforest = IsolationForestLite()
        self.vol_regime = VolatilityRegimeDetector()
        self.order_flow = OrderFlowDetector()
        self.history: deque = deque(maxlen=5000)
        log.info("ML Anomaly Detector initialized (4-model ensemble)")

    def process_tick(self, price: float, volume: int, timestamp: float) -> Dict:
        """
        Process a single market tick through the full pipeline.
        Returns dict with ensemble score, individual model scores, and anomaly flags.
        """
        fv = self.feature_extractor.extract(price, volume, timestamp)

        results = {
            "zscore": self.zscore.score(fv),
            "iforest": self.iforest.score(fv),
            "vol_regime": self.vol_regime.score(fv),
            "order_flow": self.order_flow.score(fv),
        }

        # Weighted ensemble
        ensemble_score = sum(
            results[name].score * weight
            for name, weight in self.WEIGHTS.items()
        )
        ensemble_score = round(min(1.0, ensemble_score), 4)

        # Determine primary anomaly type from highest-scoring flagged model
        primary_type = "NONE"
        max_score = 0.0
        for r in results.values():
            if r.is_anomaly and r.score > max_score:
                max_score = r.score
                primary_type = r.anomaly_type

        is_anomaly = ensemble_score > 0.45

        # Severity classification
        severity = "LOW"
        if ensemble_score > 0.8:
            severity = "CRITICAL"
        elif ensemble_score > 0.6:
            severity = "HIGH"
        elif ensemble_score > 0.45:
            severity = "MEDIUM"

        output = {
            "timestamp": float(timestamp),
            "ensemble_score": float(ensemble_score),
            "is_anomaly": bool(is_anomaly),
            "primary_type": str(primary_type),
            "severity": str(severity),
            "models": {name: {"score": float(r.score), "is_anomaly": bool(r.is_anomaly),
                              "type": str(r.anomaly_type), "details": _sanitize(r.details)}
                       for name, r in results.items()},
            "features": {
                "price_return": float(round(fv.price_return, 6)),
                "volume_ratio": float(round(fv.volume_ratio, 3)),
                "volatility_ratio": float(round(fv.volatility_ratio, 3)),
                "order_imbalance": float(round(fv.order_imbalance, 4)),
            },
        }

        self.history.append(output)
        return output

    def get_summary(self) -> Dict:
        """Get a summary of recent anomaly activity for agent context."""
        if not self.history:
            return {"total_ticks": 0, "anomalies": 0, "avg_score": 0}

        recent = list(self.history)[-100:]
        anomalies = [h for h in recent if h["is_anomaly"]]

        type_counts = {}
        for a in anomalies:
            t = a["primary_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_ticks": len(self.history),
            "recent_window": len(recent),
            "anomalies_in_window": len(anomalies),
            "anomaly_rate": float(round(len(anomalies) / len(recent), 3)),
            "avg_score": float(round(np.mean([h["ensemble_score"] for h in recent]), 4)),
            "max_score": float(round(max(h["ensemble_score"] for h in recent), 4)),
            "type_distribution": type_counts,
            "current_regime": str(recent[-1]["models"]["vol_regime"]["details"].get("regime", "UNKNOWN")),
        }
