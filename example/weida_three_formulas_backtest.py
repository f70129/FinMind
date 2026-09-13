"""
韋大「三大公式速查表」近似回測（台指期 TX 近月）

說明
----
講義參數為空白（____），本腳本採用合理預設並清楚標註。
FinMind 免費端僅有期貨日線（日盤/夜盤），無法取得逐筆／分K，因此：

1. 公式01（瞬間決策／跳空）：以「夜盤收 → 日盤開」跳空 + 日盤 OHLC 路徑近似。
2. 公式02（多空穿越）：以前一日最常被穿越價的代理（前日收盤）+ 日內高低路徑近似。
3. 公式03（動態共振）：以週線 MACD 定方向、日線 MACD 進場近似大小週期共振。

結果僅供研究，非投資建議。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "weida_backtest_results"
RAW_PATH = RESULTS_DIR / "tx_daily_raw.pkl"
POINT_VALUE = 200  # 大台指每點新台幣
CONTRACTS = 1
# 來回交易成本近似（手續費+交易稅），以點數計
ROUND_TRIP_COST_PTS = 2.0


@dataclass
class Formula01Params:
    gap_vs_avg_amp: float = 0.35  # 跳空 > 月均振幅 * 此比例
    shadow_range_max: float = 80.0  # M1-m1 近似：開盤後確認區間上限（點）
    stop_loss: float = 40.0
    take_profit_amp_mult: float = 1.2  # 停利 = 平均振幅 * 倍數
    # 日線無法精準 13:30，收盤視為時間出場


@dataclass
class Formula02Params:
    retrace_max: float = 30.0  # 拉回不超過 N 點再突破進場
    stop_loss: float = 40.0
    final_tp_amp_mult: float = 1.0  # 最終停利 = 平均振幅 * 倍數


@dataclass
class Formula03Params:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    protect_no_trend: float = 50.0
    protect_with_trend: float = 150.0


def load_near_month_sessions(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """每日選未平倉量最大合約，拆日盤 / 夜盤。"""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    def pick_near(session: str) -> pd.DataFrame:
        s = df[df["trading_session"] == session].copy()
        idx = s.groupby("date")["open_interest"].idxmax()
        out = s.loc[idx].sort_values("date").reset_index(drop=True)
        out = out.rename(
            columns={"max": "high", "min": "low", "volume": "vol"}
        )
        return out[
            ["date", "contract_date", "open", "high", "low", "close", "vol", "open_interest"]
        ]

    return pick_near("position"), pick_near("after_market")


def build_day_frame(day: pd.DataFrame, night: pd.DataFrame) -> pd.DataFrame:
    """合併夜盤收、計算跳空與振幅。"""
    n = night[["date", "close"]].rename(columns={"close": "night_close"})
    # 夜盤通常掛在「日曆日」；日盤跳空參考「前一交易日夜盤收」
    n["next_day"] = n["date"] + pd.Timedelta(days=1)
    # 對齊到下一個實際日盤交易日
    day = day.sort_values("date").reset_index(drop=True)
    night_map = night.set_index("date")["close"].sort_index()

    rows = []
    for i, r in day.iterrows():
        d = r["date"]
        # 找嚴格早於日盤日的最近夜盤收
        prev_nights = night_map[night_map.index < d]
        night_close = float(prev_nights.iloc[-1]) if len(prev_nights) else np.nan
        prev_day_close = float(day.loc[i - 1, "close"]) if i > 0 else np.nan
        rows.append(
            {
                **r.to_dict(),
                "night_close": night_close,
                "prev_day_close": prev_day_close,
            }
        )
    out = pd.DataFrame(rows)
    out["gap_vs_night"] = out["open"] - out["night_close"]
    out["gap_vs_day"] = out["open"] - out["prev_day_close"]
    out["amplitude"] = out["high"] - out["low"]
    out["avg_amp_20"] = out["amplitude"].rolling(20, min_periods=10).mean()
    out["avg_amp_month"] = out["amplitude"].rolling(20, min_periods=10).mean()
    return out.dropna(subset=["night_close", "prev_day_close", "avg_amp_month"]).reset_index(
        drop=True
    )


def simulate_path(
    entry: float,
    direction: int,
    high: float,
    low: float,
    close: float,
    stop_loss: float,
    take_profit: float,
    open_: float,
) -> Tuple[float, str]:
    """
    以日 OHLC 近似盤中路徑：開→高→低→收 與 開→低→高→收 兩路徑取較保守結果
    （先觸停損優先於停利的保守假設）。
    direction: 1 long, -1 short
    """

    def one_path(path: List[float]) -> Tuple[float, str]:
        for px in path[1:]:
            move = (px - entry) * direction
            if move <= -stop_loss:
                return entry - direction * stop_loss, "stop"
            if move >= take_profit:
                return entry + direction * take_profit, "target"
        exit_px = close
        return exit_px, "time"

    # 保守：兩種常見路徑都算，取較差損益
    path_a = [open_, high, low, close]
    path_b = [open_, low, high, close]
    e1, r1 = one_path(path_a)
    e2, r2 = one_path(path_b)
    pnl1 = (e1 - entry) * direction
    pnl2 = (e2 - entry) * direction
    if pnl1 <= pnl2:
        return e1, r1
    return e2, r2


def backtest_formula01(
    df: pd.DataFrame, p: Formula01Params, fade: bool = False
) -> pd.DataFrame:
    trades = []
    label = "01_瞬間決策_淡化" if fade else "01_瞬間決策_延續"
    for i, row in df.iterrows():
        gap = row["gap_vs_night"]
        avg_amp = row["avg_amp_month"]
        if abs(gap) <= avg_amp * p.gap_vs_avg_amp:
            continue
        # 預設：跳空方向延續；fade=True 則反向（缺口回補思維）
        # 注意：日線無法還原「影線組 M1-m1」開盤後確認，故不使用當日高低過濾
        # （避免用收盤後才知道的 high/low 造成前瞻偏差）
        direction = (-1 if gap > 0 else 1) if fade else (1 if gap > 0 else -1)

        entry = float(row["open"])
        tp = avg_amp * p.take_profit_amp_mult
        exit_px, reason = simulate_path(
            entry=entry,
            direction=direction,
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            stop_loss=p.stop_loss,
            take_profit=tp,
            open_=entry,
        )
        close_move = (float(row["close"]) - entry) * direction
        if reason == "time" and close_move <= -p.stop_loss:
            exit_px = entry - direction * p.stop_loss
            reason = "stop_close_confirm"

        pnl_pts = (exit_px - entry) * direction - ROUND_TRIP_COST_PTS
        trades.append(
            {
                "date": row["date"],
                "formula": label,
                "direction": "long" if direction > 0 else "short",
                "entry": entry,
                "exit": exit_px,
                "pnl_pts": pnl_pts,
                "reason": reason,
                "gap": gap,
                "avg_amp": avg_amp,
            }
        )
    return pd.DataFrame(trades)


def backtest_formula02(df: pd.DataFrame, p: Formula02Params) -> pd.DataFrame:
    """
    關鍵線代理 = 前日收盤（最常被穿越價的日線代理）。

    兩日結構近似「拉回後再突破」：
    - Day0：收盤在關鍵線一側，且曾穿越／測試關鍵線
    - Day1：朝突破方向再開／走勢延續，於開盤或回測關鍵線進場
    停損固定點數；停利取「下一檔前高/前低」與「振幅倍數」中較合理者。
    """
    trades = []
    for i in range(2, len(df)):
        row = df.iloc[i]
        setup = df.iloc[i - 1]
        prev = df.iloc[i - 2]
        key = float(prev["close"])
        s_o, s_h, s_l, s_c = map(
            float, (setup["open"], setup["high"], setup["low"], setup["close"])
        )
        o, h, l, c = map(float, (row["open"], row["high"], row["low"], row["close"]))
        avg_amp = float(row["avg_amp_month"])
        next_sr_up = float(prev["high"])
        next_sr_dn = float(prev["low"])

        # setup 日：測試關鍵線（拉回距離 ≤ retrace_max）
        tested = s_l <= key <= s_h
        pullback_ok = min(abs(s_l - key), abs(s_h - key), abs(s_c - key)) <= p.retrace_max
        if not (tested and pullback_ok):
            continue

        direction = None
        # 多頭結構：setup 收在關鍵線下方或附近，今日向上突破
        if s_c <= key + p.retrace_max and h > key and (o >= key - p.retrace_max or l <= key):
            if c > key or h >= key + 10:
                direction = 1
        # 空頭結構：setup 收在關鍵線上方或附近，今日向下突破
        if s_c >= key - p.retrace_max and l < key and (o <= key + p.retrace_max or h >= key):
            if c < key or l <= key - 10:
                if direction is None:
                    direction = -1
                elif s_c > key:
                    direction = -1

        if direction is None:
            continue

        # 進場：回測關鍵線則用 key，否則用開盤（已突破則追價開盤）
        if direction > 0:
            entry = key if l <= key <= h else max(o, key)
        else:
            entry = key if l <= key <= h else min(o, key)

        if direction > 0:
            sr_tp = next_sr_up - entry
        else:
            sr_tp = entry - next_sr_dn
        amp_tp = avg_amp * p.final_tp_amp_mult
        # 下一檔 SR 太近（<15點）則改用振幅停利，避免「必勝小賺」偏差
        if sr_tp >= 15:
            take_profit = min(sr_tp, amp_tp)
        else:
            take_profit = amp_tp

        exit_px, reason = simulate_path(
            entry=entry,
            direction=direction,
            high=h,
            low=l,
            close=c,
            stop_loss=p.stop_loss,
            take_profit=take_profit,
            open_=o,
        )
        pnl_pts = (exit_px - entry) * direction - ROUND_TRIP_COST_PTS
        trades.append(
            {
                "date": row["date"],
                "formula": "02_多空穿越",
                "direction": "long" if direction > 0 else "short",
                "entry": entry,
                "exit": exit_px,
                "pnl_pts": pnl_pts,
                "reason": reason,
                "key_line": key,
                "avg_amp": avg_amp,
            }
        )
    return pd.DataFrame(trades)


def macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def backtest_formula03(df: pd.DataFrame, p: Formula03Params) -> pd.DataFrame:
    """
    大週期：週線 MACD histogram 方向；小週期：日線 MACD 零軸穿越／柱狀翻轉共振。
    停損：前一波段極點（前 5 日高低）。
    保利：50 / 150 點。
    """
    d = df.copy()
    d["macd_h"] = macd_hist(d["close"], p.macd_fast, p.macd_slow, p.macd_signal)
    # 週線：以週五收合成
    weekly = (
        d.set_index("date")["close"]
        .resample("W-FRI")
        .last()
        .dropna()
        .to_frame("close")
    )
    weekly["macd_h_w"] = macd_hist(
        weekly["close"], p.macd_fast, p.macd_slow, p.macd_signal
    )
    d["week_end"] = d["date"] + pd.offsets.Week(weekday=4)
    # map each day to latest completed week macd
    w = weekly.reset_index().rename(columns={"date": "week_end"})
    d = d.merge(w[["week_end", "macd_h_w"]], on="week_end", how="left")
    d["macd_h_w"] = d["macd_h_w"].ffill()

    trades = []
    position = None
    for i in range(6, len(d)):
        row = d.iloc[i]
        prev = d.iloc[i - 1]
        if pd.isna(row["macd_h_w"]) or pd.isna(row["macd_h"]):
            continue

        big_dir = 1 if row["macd_h_w"] > 0 else -1
        # 共振：日線柱由負轉正（或多）且大週期同向
        cross_up = prev["macd_h"] <= 0 < row["macd_h"]
        cross_dn = prev["macd_h"] >= 0 > row["macd_h"]

        if position is None:
            if big_dir > 0 and cross_up:
                sl = float(d.iloc[i - 5 : i]["low"].min())
                position = {
                    "date": row["date"],
                    "direction": 1,
                    "entry": float(row["close"]),
                    "sl": sl,
                    "trend_cont": abs(row["macd_h_w"]) > abs(d.iloc[i - 5 : i]["macd_h_w"].mean()),
                }
            elif big_dir < 0 and cross_dn:
                sl = float(d.iloc[i - 5 : i]["high"].max())
                position = {
                    "date": row["date"],
                    "direction": -1,
                    "entry": float(row["close"]),
                    "sl": sl,
                    "trend_cont": abs(row["macd_h_w"]) > abs(d.iloc[i - 5 : i]["macd_h_w"].mean()),
                }
            continue

        # manage open position on subsequent days
        direction = position["direction"]
        entry = position["entry"]
        o, h, l, c = map(float, (row["open"], row["high"], row["low"], row["close"]))
        protect = (
            p.protect_with_trend if position["trend_cont"] else p.protect_no_trend
        )
        stop_dist = abs(entry - position["sl"])
        stop_dist = max(stop_dist, 20.0)

        # 保利：獲利達門檻後，停損上移至成本（保本）或保利水位
        be_level = protect  # 觸及後鎖定至少 0 點（簡化：出場保利水位的一半）
        # 路徑模擬：停損為前柱群極點距離；觸及保利後改保本
        exit_px = None
        reason = None

        # 先檢查原始停損
        def hit_stop(px_low, px_high):
            if direction > 0 and px_low <= position["sl"]:
                return position["sl"], "stop_swing"
            if direction < 0 and px_high >= position["sl"]:
                return position["sl"], "stop_swing"
            return None, None

        # 有利方向極值
        favor = (h - entry) if direction > 0 else (entry - l)
        adverse_exit, adverse_reason = hit_stop(l, h)

        if favor >= protect:
            # 保利啟動：收盤若回到成本則出；當日若掃到成本也出
            be_price = entry
            if direction > 0 and l <= be_price:
                exit_px, reason = be_price, f"protect_{int(protect)}"
            elif direction < 0 and h >= be_price:
                exit_px, reason = be_price, f"protect_{int(protect)}"
            else:
                # 續抱至收盤，若 MACD 反向則出
                if (direction > 0 and row["macd_h"] < 0) or (
                    direction < 0 and row["macd_h"] > 0
                ):
                    exit_px, reason = c, "macd_exit"
                else:
                    # 未出：更新浮盈，繼續
                    pass
        elif adverse_exit is not None:
            exit_px, reason = adverse_exit, adverse_reason

        # MACD 反向出場（無保利時）
        if exit_px is None:
            if (direction > 0 and row["macd_h"] < 0 and prev["macd_h"] >= 0) or (
                direction < 0 and row["macd_h"] > 0 and prev["macd_h"] <= 0
            ):
                exit_px, reason = c, "macd_exit"

        if exit_px is None:
            continue

        pnl_pts = (exit_px - entry) * direction - ROUND_TRIP_COST_PTS
        trades.append(
            {
                "date": position["date"],
                "exit_date": row["date"],
                "formula": "03_動態共振",
                "direction": "long" if direction > 0 else "short",
                "entry": entry,
                "exit": exit_px,
                "pnl_pts": pnl_pts,
                "reason": reason,
            }
        )
        position = None

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, name: str) -> Dict:
    if trades is None or len(trades) == 0:
        return {
            "strategy": name,
            "trades": 0,
            "win_rate": None,
            "total_pnl_pts": 0.0,
            "total_pnl_twd": 0.0,
            "avg_pnl_pts": None,
            "max_dd_pts": None,
            "profit_factor": None,
        }
    pnl = trades["pnl_pts"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    equity = pnl.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    gp = wins.sum()
    gl = -losses.sum()
    return {
        "strategy": name,
        "trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "total_pnl_pts": float(pnl.sum()),
        "total_pnl_twd": float(pnl.sum() * POINT_VALUE * CONTRACTS),
        "avg_pnl_pts": float(pnl.mean()),
        "max_dd_pts": float(dd.min()) if len(dd) else 0.0,
        "profit_factor": float(gp / gl) if gl > 0 else None,
        "long_trades": int((trades["direction"] == "long").sum()),
        "short_trades": int((trades["direction"] == "short").sum()),
    }


def plot_equity(trades_map: Dict[str, pd.DataFrame], out_path: Path) -> None:
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(11, 6))
    for name, tr in trades_map.items():
        if tr is None or len(tr) == 0:
            continue
        eq = tr["pnl_pts"].cumsum()
        x = pd.to_datetime(tr["date"])
        plt.plot(x, eq, label=name, linewidth=1.8)
    plt.axhline(0, color="#888", linewidth=0.8)
    plt.title("Weida 3 Formulas Approx Backtest — Cum. Points (TX near-month x1)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (points)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        raw = pd.read_pickle(RAW_PATH)
    else:
        from FinMind.data import DataLoader

        raw = DataLoader().taiwan_futures_daily(
            futures_id="TX", start_date="2022-01-01", end_date="2025-12-31"
        )
        raw.to_pickle(RAW_PATH)

    day, night = load_near_month_sessions(raw)
    df = build_day_frame(day, night)
    # 回測區間：留足 rolling
    df = df[df["date"] >= "2022-03-01"].reset_index(drop=True)

    p1 = Formula01Params()
    p2 = Formula02Params()
    p3 = Formula03Params()

    t1 = backtest_formula01(df, p1, fade=False)
    t1_fade = backtest_formula01(df, p1, fade=True)
    t2 = backtest_formula02(df, p2)
    t3 = backtest_formula03(df, p3)

    summaries = [
        summarize(t1, "01_瞬間決策_延續"),
        summarize(t1_fade, "01_瞬間決策_淡化"),
        summarize(t2, "02_多空穿越"),
        summarize(t3, "03_動態共振"),
    ]
    # 主組合：延續版公式01 + 02 + 03
    all_tr = pd.concat([t1, t2, t3], ignore_index=True)
    if len(all_tr):
        all_tr = all_tr.sort_values("date").reset_index(drop=True)
    summaries.append(summarize(all_tr, "ALL_合計(01延續+02+03)"))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    t1.to_csv(RESULTS_DIR / "trades_formula01.csv", index=False)
    t1_fade.to_csv(RESULTS_DIR / "trades_formula01_fade.csv", index=False)
    t2.to_csv(RESULTS_DIR / "trades_formula02.csv", index=False)
    t3.to_csv(RESULTS_DIR / "trades_formula03.csv", index=False)

    params = {
        "instrument": "TX near-month",
        "session": "day (gap vs prior night close)",
        "period": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "days": int(len(df)),
        },
        "point_value_twd": POINT_VALUE,
        "contracts": CONTRACTS,
        "formula01": asdict(p1),
        "formula02": asdict(p2),
        "formula03": asdict(p3),
        "limitations": [
            "講義空白參數以預設值填入，非課程官方數字",
            "無分K/逐筆，路徑與13:30出場為日OHLC近似",
            "公式01影線組M1-m1因無分K而省略（避免用當日高低造成前瞻偏差）",
            "公式02關鍵線以前日收盤代理「最多穿越價」",
            "公式03以週/日MACD代理大小分時共振，非波浪理論完整實作",
        ],
    }
    (RESULTS_DIR / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_equity(
        {
            "01_gap_continue": t1,
            "01_gap_fade": t1_fade,
            "02_cross": t2,
            "03_macd_resonance": t3,
        },
        RESULTS_DIR / "equity_curves.png",
    )

    # 簡易敏感度：公式01 跳空門檻
    sens_rows = []
    for gap_r in [0.2, 0.35, 0.5, 0.7]:
        for sl in [30, 40, 60]:
            for tp_m in [0.8, 1.2, 1.6]:
                tr = backtest_formula01(
                    df,
                    Formula01Params(
                        gap_vs_avg_amp=gap_r, stop_loss=sl, take_profit_amp_mult=tp_m
                    ),
                )
                s = summarize(tr, f"gap{gap_r}_sl{sl}_tp{tp_m}")
                s.update({"gap_r": gap_r, "sl": sl, "tp_m": tp_m})
                sens_rows.append(s)
    sens = pd.DataFrame(sens_rows).sort_values("total_pnl_pts", ascending=False)
    sens.to_csv(RESULTS_DIR / "formula01_sensitivity.csv", index=False)

    print("=== 參數 ===")
    print(json.dumps(params, ensure_ascii=False, indent=2))
    print("\n=== 績效摘要 ===")
    print(summary_df.to_string(index=False))
    print("\n=== 公式01敏感度 Top5 ===")
    print(
        sens[
            [
                "gap_r",
                "sl",
                "tp_m",
                "trades",
                "win_rate",
                "total_pnl_pts",
                "max_dd_pts",
                "profit_factor",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
