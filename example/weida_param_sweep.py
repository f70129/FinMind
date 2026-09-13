"""韋大公式參數暴力搜尋：跳空網格 + 穿越/共振 1分&5分。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "weida_backtest_results"
SEC5 = ROOT / "taiex_5s"
OUT = ROOT / "sweep_v4"
COST = 2.0


@dataclass
class EP:
    sl: float
    tp: float


def load_ohlc(rule: str) -> pd.DataFrame:
    frames = []
    for fp in sorted(SEC5.glob("*.pkl")):
        raw = pd.read_pickle(fp)
        raw["date"] = pd.to_datetime(raw["date"])
        s = raw.set_index("date")["TAIEX"].sort_index()
        ohlc = s.resample(rule).ohlc().dropna()
        if ohlc.empty:
            continue
        ohlc["session"] = ohlc.index.normalize()
        frames.append(ohlc)
    return pd.concat(frames).sort_index()


def ensure_session(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "session" in df.columns:
        return df
    if "session_date" in df.columns:
        return df.rename(columns={"session_date": "session"})
    df["session"] = df.index.normalize()
    return df


def macd_hist(close: pd.Series) -> pd.Series:
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()


def simulate(entry: float, d: int, bars: pd.DataFrame, i0: int, ep: EP) -> Tuple[float, str]:
    for j in range(i0 + 1, len(bars)):
        r = bars.iloc[j]
        h, l, c = float(r["high"]), float(r["low"]), float(r["close"])
        fav = (h - entry) if d > 0 else (entry - l)
        adv = (entry - l) if d > 0 else (h - entry)
        if adv >= ep.sl:
            return -ep.sl - COST, "stop"
        if fav >= ep.tp:
            return ep.tp - COST, "tp"
        ts = bars.index[j]
        if hasattr(ts, "time") and ts.time() >= pd.Timestamp("13:25").time():
            return (c - entry) * d - COST, "time"
    c = float(bars.iloc[-1]["close"])
    return (c - entry) * d - COST, "time"


def make_gaps() -> pd.DataFrame:
    rows = []
    for fp in sorted(SEC5.glob("*.pkl")):
        raw = pd.read_pickle(fp)
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date")
        if raw.empty:
            continue
        d = raw["date"].iloc[0].normalize()
        close = float(raw["TAIEX"].iloc[-1])
        w = raw[
            (raw["date"].dt.time >= pd.Timestamp("09:00:30").time())
            & (raw["date"].dt.time <= pd.Timestamp("09:01:00").time())
        ]
        op = float(w["TAIEX"].mean()) if len(w) else float(raw["TAIEX"].iloc[1:12].mean())
        rows.append({"session": d, "open_px": op, "close_px": close})
    g = pd.DataFrame(rows).sort_values("session")
    g["gap"] = g["open_px"] - g["close_px"].shift(1)
    return g.set_index("session")


def most_crossed(day: pd.DataFrame, step: float = 5.0) -> float:
    if len(day) < 2:
        return float(day["close"].iloc[-1])
    lo = np.floor(day["low"].min() / step) * step
    hi = np.ceil(day["high"].max() / step) * step
    levels = np.arange(lo, hi + step, step)
    counts = np.zeros(len(levels))
    closes = day["close"].to_numpy()
    for a, b in zip(closes[:-1], closes[1:]):
        x, y = (a, b) if a <= b else (b, a)
        counts += ((levels >= x) & (levels <= y)).astype(float)
    return float(levels[int(np.argmax(counts))])


def score_sigs(sigs: List[dict], ep: EP) -> Dict:
    if not sigs:
        return dict(
            trades=0, total_pnl=0.0, avg_pnl=None, win_rate=None, pf=None,
            max_dd=0.0, tp_rate=None, stop_rate=None, score=-1e18,
        )
    pnls, reasons = [], []
    for s in sigs:
        pnl, reason = simulate(s["entry"], s["dir"], s["bars"], s["i"], ep)
        pnls.append(pnl)
        reasons.append(reason)
    p = pd.Series(pnls, dtype=float)
    eq = p.cumsum()
    dd = float((eq - eq.cummax()).min())
    gp = float(p[p > 0].sum())
    gl = float((-p[p <= 0]).sum())
    pf = (gp / gl) if gl > 0 else (10.0 if gp > 0 else 0.0)
    total = float(p.sum())
    n = len(p)
    score = total + min(dd, 0) * 0.5 + (pf - 1) * 200
    if n < 40:
        score -= (40 - n) * 30
    return dict(
        trades=n,
        total_pnl=total,
        avg_pnl=float(p.mean()),
        win_rate=float((p > 0).mean()),
        pf=pf,
        max_dd=dd,
        tp_rate=float(sum(r == "tp" for r in reasons) / n),
        stop_rate=float(sum(r == "stop" for r in reasons) / n),
        score=score,
    )


def f01_sigs(bars: pd.DataFrame, gaps: pd.DataFrame, gm: float) -> List[dict]:
    daily = bars.groupby("session").agg(high=("high", "max"), low=("low", "min"))
    daily["amp"] = daily["high"] - daily["low"]
    daily["avg_amp"] = daily["amp"].rolling(20, min_periods=10).mean()
    daily = daily.join(gaps[["gap"]])
    by = {d: g for d, g in bars.groupby("session")}
    out = []
    for d, row in daily.dropna(subset=["gap", "avg_amp"]).iterrows():
        if abs(row["gap"]) <= row["avg_amp"] * gm:
            continue
        day = by.get(d)
        if day is None or len(day) < 3:
            continue
        i = min(1, len(day) - 1)
        out.append(dict(
            date=d,
            dir=1 if row["gap"] > 0 else -1,
            entry=float(day.iloc[i]["open"]),
            i=i,
            bars=day,
        ))
    return out


def f02_sigs(bars: pd.DataFrame, retrace: float) -> List[dict]:
    days = sorted(bars["session"].unique())
    by = {d: g for d, g in bars.groupby("session")}
    out = []
    for i in range(1, len(days)):
        key = most_crossed(by[days[i - 1]])
        day = by[days[i]]
        touched = False
        for j in range(len(day)):
            h, l = float(day.iloc[j]["high"]), float(day.iloc[j]["low"])
            if not touched:
                if l <= key <= h and min(key - l, h - key) <= retrace:
                    touched = True
                continue
            direction = entry = None
            if h >= key + 5 and float(day.iloc[j]["close"]) > key:
                direction, entry = 1, max(key, float(day.iloc[j]["open"]))
            elif l <= key - 5 and float(day.iloc[j]["close"]) < key:
                direction, entry = -1, min(key, float(day.iloc[j]["open"]))
            if direction is None:
                continue
            out.append(dict(date=days[i], dir=direction, entry=entry, i=j, bars=day))
            break
    return out


def f03_same(bars: pd.DataFrame) -> List[dict]:
    b = bars.copy()
    b["mh"] = macd_hist(b["close"])
    out = []
    for d, g in b.groupby("session"):
        g = g.copy()
        if len(g) < 35:
            continue
        for j in range(1, len(g)):
            a, c = g.iloc[j - 1]["mh"], g.iloc[j]["mh"]
            if pd.isna(a) or pd.isna(c):
                continue
            if a <= 0 < c:
                direction = 1
            elif a >= 0 > c:
                direction = -1
            else:
                continue
            out.append(dict(date=d, dir=direction, entry=float(g.iloc[j]["close"]), i=j, bars=g))
            break
    return out


def f03_multi(fast: pd.DataFrame, slow: pd.DataFrame) -> List[dict]:
    left = fast.copy()
    left["mh"] = macd_hist(fast["close"])
    left["dt"] = left.index
    left = left.reset_index(drop=True).sort_values("dt")
    right = slow.copy()
    right["mh_s"] = macd_hist(slow["close"])
    right["dt"] = right.index
    right = right.reset_index(drop=True)[["dt", "mh_s"]].sort_values("dt")
    m = pd.merge_asof(left, right, on="dt", direction="backward").set_index("dt").sort_index()
    out = []
    for d, g in m.groupby("session"):
        g = g.copy()
        if len(g) < 35:
            continue
        for j in range(1, len(g)):
            big = g.iloc[j]["mh_s"]
            a, c = g.iloc[j - 1]["mh"], g.iloc[j]["mh"]
            if pd.isna(big) or pd.isna(a) or pd.isna(c):
                continue
            big_d = 1 if big > 0 else -1
            if big_d > 0 and a <= 0 < c:
                direction = 1
            elif big_d < 0 and a >= 0 > c:
                direction = -1
            else:
                continue
            out.append(dict(date=d, dir=direction, entry=float(g.iloc[j]["close"]), i=j, bars=g))
            break
    return out


def sweep(sigs: List[dict], sls, tps, meta: dict) -> pd.DataFrame:
    rows = []
    for sl, tp in product(sls, tps):
        if tp <= sl:
            continue
        st = score_sigs(sigs, EP(sl, tp))
        rows.append({**meta, "sl": sl, "tp": tp, **st})
    return pd.DataFrame(rows)


def equity(sigs: List[dict], ep: EP) -> pd.Series:
    rows = [
        dict(date=s["date"], pnl=simulate(s["entry"], s["dir"], s["bars"], s["i"], ep)[0])
        for s in sigs
    ]
    tr = pd.DataFrame(rows)
    if tr.empty:
        return pd.Series(dtype=float)
    return tr.set_index("date")["pnl"].cumsum()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Load bars...")
    c2 = ROOT / "intraday_v2"
    bars5 = ensure_session(
        pd.read_pickle(c2 / "bars_5min.pkl") if (c2 / "bars_5min.pkl").exists() else load_ohlc("5min")
    )
    bars30 = ensure_session(
        pd.read_pickle(c2 / "bars_30min.pkl") if (c2 / "bars_30min.pkl").exists() else load_ohlc("30min")
    )
    p1 = OUT / "bars_1min.pkl"
    if p1.exists():
        bars1 = ensure_session(pd.read_pickle(p1))
    else:
        print("Resample 1min...")
        bars1 = load_ohlc("1min")
        bars1.to_pickle(p1)
    print("days", bars5["session"].nunique())

    gaps = make_gaps()
    tf = {"1min": bars1, "5min": bars5, "30min": bars30}

    print("=== F01 ===")
    parts = []
    for name, bars in tf.items():
        for gm in [0.2, 0.3, 0.35, 0.5, 0.7]:
            sigs = f01_sigs(bars, gaps, gm)
            print(f"  {name} gm={gm} n={len(sigs)}")
            parts.append(
                sweep(
                    sigs,
                    [30, 40, 50, 60, 80, 100, 120],
                    [80, 100, 120, 150, 200, 250, 300, 400, 480],
                    dict(formula="01", tf=name, gap_mult=gm),
                )
            )
    f01 = pd.concat(parts, ignore_index=True).sort_values(["score", "total_pnl"], ascending=False)
    f01.to_csv(OUT / "f01_gap_sweep.csv", index=False)

    print("=== F02 ===")
    parts = []
    f02_cache = {}
    for name, bars in [("1min", bars1), ("5min", bars5)]:
        for retr in [20, 30, 50, 80]:
            sigs = f02_sigs(bars, retr)
            f02_cache[(name, retr)] = sigs
            print(f"  {name} retr={retr} n={len(sigs)}")
            parts.append(
                sweep(
                    sigs,
                    [40, 50, 60, 80],
                    [100, 120, 150, 200, 250],
                    dict(formula="02", tf=name, retrace=retr),
                )
            )
    f02 = pd.concat(parts, ignore_index=True).sort_values(["score", "total_pnl"], ascending=False)
    f02.to_csv(OUT / "f02_cross_sweep.csv", index=False)

    print("=== F03 ===")
    setups = {
        "1min_only": f03_same(bars1),
        "5min_only": f03_same(bars5),
        "5dir_1entry": f03_multi(bars1, bars5),
        "30dir_5entry": f03_multi(bars5, bars30),
    }
    parts = []
    for name, sigs in setups.items():
        print(f"  {name} n={len(sigs)}")
        parts.append(
            sweep(
                sigs,
                [40, 50, 60, 80],
                [100, 120, 150, 200, 250],
                dict(formula="03", setup=name),
            )
        )
    f03 = pd.concat(parts, ignore_index=True).sort_values(["score", "total_pnl"], ascending=False)
    f03.to_csv(OUT / "f03_resonance_sweep.csv", index=False)

    top = {
        "f01_top15": f01.head(15).to_dict("records"),
        "f02_top10": f02.head(10).to_dict("records"),
        "f03_top10": f03.head(10).to_dict("records"),
        "baseline_sl50_tp150": {
            "f01_30_gm035": f01.query("tf=='30min' and gap_mult==0.35 and sl==50 and tp==150").head(1).to_dict("records"),
            "f02_5_r30": f02.query("tf=='5min' and retrace==30 and sl==50 and tp==150").head(1).to_dict("records"),
            "f02_1_r30": f02.query("tf=='1min' and retrace==30 and sl==50 and tp==150").head(1).to_dict("records"),
            "f03_5dir1": f03.query("setup=='5dir_1entry' and sl==50 and tp==150").head(1).to_dict("records"),
            "f03_1only": f03.query("setup=='1min_only' and sl==50 and tp==150").head(1).to_dict("records"),
            "f03_5only": f03.query("setup=='5min_only' and sl==50 and tp==150").head(1).to_dict("records"),
        },
    }
    (OUT / "tops.json").write_text(json.dumps(top, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    plt.figure(figsize=(12, 6))
    b1 = f01.iloc[0]
    eq = equity(f01_sigs(tf[b1["tf"]], gaps, float(b1["gap_mult"])), EP(float(b1["sl"]), float(b1["tp"])))
    if len(eq):
        plt.plot(eq.index, eq.values, label=f"F01 {b1['tf']} gm{b1['gap_mult']} SL{b1['sl']}/TP{b1['tp']}", lw=2)
    b2 = f02.iloc[0]
    eq = equity(f02_cache[(b2["tf"], float(b2["retrace"]))], EP(float(b2["sl"]), float(b2["tp"])))
    if len(eq):
        plt.plot(eq.index, eq.values, label=f"F02 {b2['tf']} r{b2['retrace']} SL{b2['sl']}/TP{b2['tp']}", lw=1.8)
    b3 = f03.iloc[0]
    eq = equity(setups[b3["setup"]], EP(float(b3["sl"]), float(b3["tp"])))
    if len(eq):
        plt.plot(eq.index, eq.values, label=f"F03 {b3['setup']} SL{b3['sl']}/TP{b3['tp']}", lw=1.8)
    plt.axhline(0, color="#888", lw=0.8)
    plt.title("Weida brute-force sweep — best configs")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "best_equity.png", dpi=140)
    plt.close()

    print("\n=== F01 TOP 12 ===")
    print(f01[["tf", "gap_mult", "sl", "tp", "trades", "win_rate", "total_pnl", "max_dd", "pf", "tp_rate", "score"]].head(12).to_string(index=False))
    print("\n=== F02 TOP 10 ===")
    print(f02[["tf", "retrace", "sl", "tp", "trades", "win_rate", "total_pnl", "max_dd", "pf", "tp_rate", "score"]].head(10).to_string(index=False))
    print("\n=== F03 TOP 10 ===")
    print(f03[["setup", "sl", "tp", "trades", "win_rate", "total_pnl", "max_dd", "pf", "tp_rate", "score"]].head(10).to_string(index=False))
    print("\nBaseline SL50/TP150 refs:")
    print(json.dumps(top["baseline_sl50_tp150"], ensure_ascii=False, indent=2, default=str))
    print("done", OUT)


if __name__ == "__main__":
    main()
