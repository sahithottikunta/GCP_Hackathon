"""
SENTINEL — Multi-Ticker Synthetic Market Data Stream
Generates realistic tick-level market data for multiple symbols.
Feeds the ML pipeline and broadcasts to connected clients.
"""

from __future__ import annotations
import asyncio
import time
import math
import random
from typing import Optional, Callable, Awaitable, Dict, List
from collections import deque
from models.anomaly_detector import AnomalyDetectorML
from utils import get_logger

log = get_logger("data_stream")

# ── Ticker Configurations ─────────────────────────────
TICKER_CONFIGS = {
    "AAPL": {
        "name": "Apple Inc.",
        "price": 227.48,
        "base_volatility": 0.012,
        "avg_volume": 55_000_000,
        "sector": "tech",
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "price": 272.15,
        "base_volatility": 0.028,
        "avg_volume": 85_000_000,
        "sector": "tech",
    },
    "SPY": {
        "name": "SPDR S&P 500 ETF",
        "price": 570.32,
        "base_volatility": 0.008,
        "avg_volume": 72_000_000,
        "sector": "index",
    },
    "QQQ": {
        "name": "Invesco QQQ (Nasdaq-100)",
        "price": 490.60,
        "base_volatility": 0.010,
        "avg_volume": 42_000_000,
        "sector": "index",
    },
    "ES": {
        "name": "E-mini S&P 500 Futures",
        "price": 5715.25,
        "base_volatility": 0.009,
        "avg_volume": 1_800_000,
        "sector": "futures",
    },
    "SPX": {
        "name": "S&P 500 Index",
        "price": 5712.50,
        "base_volatility": 0.008,
        "avg_volume": 0,
        "sector": "index",
    },
    "NVDA": {
        "name": "NVIDIA Corp.",
        "price": 131.88,
        "base_volatility": 0.022,
        "avg_volume": 230_000_000,
        "sector": "tech",
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "price": 205.74,
        "base_volatility": 0.014,
        "avg_volume": 44_000_000,
        "sector": "tech",
    },
    "META": {
        "name": "Meta Platforms Inc.",
        "price": 612.77,
        "base_volatility": 0.016,
        "avg_volume": 18_000_000,
        "sector": "tech",
    },
    "MSFT": {
        "name": "Microsoft Corp.",
        "price": 410.35,
        "base_volatility": 0.011,
        "avg_volume": 22_000_000,
        "sector": "tech",
    },
}


class TickerState:
    """Holds live state for a single ticker."""

    def __init__(self, symbol: str, cfg: dict):
        self.symbol = symbol
        self.name = cfg["name"]
        self.price = cfg["price"]
        self.prev_close = cfg["price"]
        self.volume = cfg["avg_volume"] or 1_000_000
        self.volatility = cfg["base_volatility"]
        self.base_volatility = cfg["base_volatility"]
        self.avg_volume = cfg["avg_volume"] or 1_000_000
        self.sector = cfg["sector"]
        self.tick_count = 0
        self.history: deque = deque(maxlen=500)


class MarketDataStream:
    """
    Multi-ticker synthetic market data generator with:
    - Per-symbol GBM price dynamics
    - Cross-sector correlation (market-wide shocks)
    - Independent + correlated anomaly injection
    - ML anomaly scoring on the primary ticker
    """

    def __init__(self):
        self.tickers: Dict[str, TickerState] = {}
        for sym, cfg in TICKER_CONFIGS.items():
            self.tickers[sym] = TickerState(sym, cfg)

        self.symbols = list(self.tickers.keys())
        self.primary_symbol = "SPY"
        self.tick_count = 0
        self.running = False

        self.detector = AnomalyDetectorML()
        self.price_history: deque = deque(maxlen=2000)
        self.anomaly_history: deque = deque(maxlen=500)
        self._callbacks: list[Callable] = []
        self._anomaly_callbacks: list[Callable] = []

    def on_tick(self, callback: Callable[[dict], Awaitable]):
        self._callbacks.append(callback)

    def on_anomaly(self, callback: Callable[[dict], Awaitable]):
        self._anomaly_callbacks.append(callback)

    # ── Tick generation ───────────────────────────────

    def _market_shock(self) -> float:
        """Occasional market-wide shock that correlates all tickers."""
        if random.random() < 0.02:
            return random.choice([-1, 1]) * random.gauss(0, 0.003)
        return 0.0

    def _generate_tick_for(self, ts: TickerState, market_factor: float) -> dict:
        """Generate one tick for a single ticker."""
        ts.tick_count += 1
        now = time.time()

        # Volatility dynamics (mean-reverting)
        ts.volatility += 0.1 * (ts.base_volatility - ts.volatility) + random.gauss(0, 0.001)
        ts.volatility = max(0.003, min(0.10, ts.volatility))

        # Anomaly injection (4% chance per ticker)
        shock = 0.0
        vol_shock = 1.0
        if random.random() < 0.04:
            shock = random.choice([-1, 1]) * random.uniform(0.002, 0.008) * ts.price
            vol_shock = random.uniform(2.0, 5.0)

        # Price: GBM + market correlation + idiosyncratic
        drift = 0.00005 * math.sin(self.tick_count / 300)
        noise = random.gauss(0, 1) * ts.volatility * ts.price
        correlated = market_factor * ts.price
        ts.price = max(ts.price * 0.5, ts.price + drift * ts.price + noise + shock + correlated)

        # Volume
        if ts.avg_volume > 0:
            base_vol = ts.avg_volume * (0.85 + random.random() * 0.3)
            ts.volume = int(max(50_000, base_vol * vol_shock / 100))
        else:
            ts.volume = 0

        # OHLC
        high = ts.price + abs(random.gauss(0, 1)) * ts.volatility * ts.price
        low = ts.price - abs(random.gauss(0, 1)) * ts.volatility * ts.price
        opn = ts.price + random.gauss(0, 0.3)

        # Day change
        change = ts.price - ts.prev_close
        change_pct = (change / ts.prev_close) * 100 if ts.prev_close else 0.0

        tick = {
            "timestamp": now,
            "symbol": ts.symbol,
            "name": ts.name,
            "price": round(ts.price, 2),
            "open": round(opn, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "volume": ts.volume,
            "volatility": round(ts.volatility * 100, 3),
            "change": round(change, 2),
            "change_pct": round(change_pct, 3),
            "tick_number": self.tick_count,
            "sector": ts.sector,
        }

        ts.history.append(tick)
        return tick

    def _generate_all_ticks(self) -> List[dict]:
        """Generate ticks for ALL tickers in one cycle."""
        self.tick_count += 1
        market_factor = self._market_shock()
        ticks = []
        for sym in self.symbols:
            ts = self.tickers[sym]
            tick = self._generate_tick_for(ts, market_factor)
            ticks.append(tick)
        return ticks

    # ── Main loop ─────────────────────────────────────

    async def start(self, interval_ms: int = 500):
        """Start the data stream loop."""
        self.running = True
        log.info(f"Market data stream started — {len(self.symbols)} tickers @ {interval_ms}ms")

        while self.running:
            all_ticks = self._generate_all_ticks()

            # Pick primary ticker for ML scoring
            primary_tick = next((t for t in all_ticks if t["symbol"] == self.primary_symbol), all_ticks[0])

            # ML scoring on primary
            ml_result = self.detector.process_tick(
                primary_tick["price"], primary_tick["volume"], primary_tick["timestamp"]
            )
            primary_tick["anomaly_score"] = float(ml_result["ensemble_score"])
            primary_tick["is_anomaly"] = bool(ml_result["is_anomaly"])
            primary_tick["anomaly_type"] = str(ml_result["primary_type"])
            primary_tick["severity"] = str(ml_result["severity"])
            primary_tick["model_scores"] = {
                k: float(v["score"]) for k, v in ml_result["models"].items()
            }

            # Simulated score for non-primary tickers
            for t in all_ticks:
                if t["symbol"] != self.primary_symbol:
                    t["anomaly_score"] = round(
                        random.uniform(0, 0.3) if random.random() > 0.05 else random.uniform(0.4, 0.95), 3
                    )
                    t["is_anomaly"] = t["anomaly_score"] > 0.5
                    t["anomaly_type"] = "NONE"
                    t["severity"] = "LOW"
                    t["model_scores"] = {}

            self.price_history.append(primary_tick)

            # Build combined payload
            combined = {
                "tickers": all_ticks,
                "primary": primary_tick,
                "tick_number": self.tick_count,
            }

            for cb in self._callbacks:
                try:
                    await cb(combined)
                except Exception as e:
                    log.error(f"Tick callback error: {e}")

            if ml_result["is_anomaly"]:
                self.anomaly_history.append(ml_result)
                for cb in self._anomaly_callbacks:
                    try:
                        await cb(ml_result)
                    except Exception as e:
                        log.error(f"Anomaly callback error: {e}")

            await asyncio.sleep(interval_ms / 1000.0)

    def stop(self):
        self.running = False
        log.info("Market data stream stopped")

    def get_snapshot(self) -> dict:
        """Current market state for agent context — includes all tickers."""
        if not self.price_history:
            return {"status": "no_data"}

        ticker_summaries = {}
        for sym, ts in self.tickers.items():
            if ts.history:
                recent = list(ts.history)[-30:]
                prices = [t["price"] for t in recent]
                ticker_summaries[sym] = {
                    "name": ts.name,
                    "current_price": recent[-1]["price"],
                    "change": recent[-1].get("change", 0),
                    "change_pct": recent[-1].get("change_pct", 0),
                    "high_30": round(max(prices), 2),
                    "low_30": round(min(prices), 2),
                    "volatility": recent[-1]["volatility"],
                    "volume": recent[-1]["volume"],
                }

        recent_primary = list(self.price_history)[-50:]
        prices = [t["price"] for t in recent_primary]
        volumes = [t["volume"] for t in recent_primary]

        return {
            "tickers": ticker_summaries,
            "primary_symbol": self.primary_symbol,
            "current_price": recent_primary[-1]["price"],
            "current_volume": recent_primary[-1]["volume"],
            "price_range_50": {"min": round(min(prices), 2), "max": round(max(prices), 2)},
            "avg_volume_50": int(sum(volumes) / len(volumes)),
            "current_volatility": recent_primary[-1]["volatility"],
            "total_ticks": self.tick_count,
            "recent_anomalies": len([a for a in list(self.anomaly_history)[-20:]]),
            "ml_summary": self.detector.get_summary(),
        }


# Singleton
market_stream = MarketDataStream()
