# -*- coding: utf-8 -*-
# process_report.py — генерит картинки и CSV для поиска узких мест

import argparse
import math
import os
import sys
from typing import Dict, Tuple, Optional
from collections import Counter, defaultdict
import glob
import difflib
from statistics import median
from scipy.stats import kendalltau
import numpy as np

import pandas as pd
import pm4py
from pm4py.algo.discovery.dfg import algorithm as dfg_algo
from graphviz import Digraph

# -------------------------------
# Вспомогательные функции
# -------------------------------

def _humanize_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    seconds = float(seconds)
    if seconds < 1:
        # показать миллисекунды при очень малых значениях
        return f"{seconds * 1000:.0f} ms"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and len(parts) < 3:
        parts.append(f"{minutes}m")
    if sec and len(parts) < 3:
        parts.append(f"{sec}s")
    return " ".join(parts) if parts else f"{int(seconds)}s"


def _format_dataframe(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    ts_col: str,
    ts_format: Optional[str]
) -> pd.DataFrame:
    # Парсинг времени
    if ts_format:
        df[ts_col] = pd.to_datetime(df[ts_col], format=ts_format, errors="coerce")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    # Отфильтровать события без времени или активности
    df = df.dropna(subset=[case_col, activity_col, ts_col]).copy()

    # Нормализация в формат pm4py
    try:
        formatted_df = pm4py.format_dataframe(
            df,
            case_id=case_col,
            activity_key=activity_col,
            timestamp_key=ts_col
        )
    except Exception:
        # Если в текущей версии pm4py отсутствует format_dataframe
        formatted_df = df.rename(columns={
            case_col: "case:concept:name",
            activity_col: "concept:name",
            ts_col: "time:timestamp",
        }).copy()
    return formatted_df


def _to_event_log(formatted_df: pd.DataFrame):
    # Конвертация в EventLog (некоторые функции dfg работают и с DF, но лог надёжнее)
    try:
        from pm4py.objects.conversion.log import converter as log_converter
        log = log_converter.apply(formatted_df)
        return log
    except Exception:
        try:
            # High-level API в некоторых версиях
            return pm4py.convert_to_event_log(formatted_df)
        except Exception:
            # Как фолбэк — вернём сам DataFrame (dfg.apply умеет работать с DF)
            return formatted_df


def _discover_frequency_dfg(log_like) -> Dict[Tuple[str, str], int]:
    try:
        dfg = dfg_algo.apply(log_like, variant=dfg_algo.Variants.FREQUENCY)
        # dfg — dict с ключами (act_from, act_to) -> count
        return dict(dfg)
    except Exception:
        # Фолбэк: вручную по DataFrame
        if isinstance(log_like, pd.DataFrame):
            df = log_like
        else:
            # Если это не DF, возвращаем пусто
            return {}
        df = df.sort_values(["case:concept:name", "time:timestamp"])  # type: ignore
        counts: Dict[Tuple[str, str], int] = Counter()
        for _, g in df.groupby("case:concept:name"):  # type: ignore
            acts = g["concept:name"].tolist()  # type: ignore
            for i in range(1, len(acts)):
                counts[(acts[i-1], acts[i])] += 1
        return dict(counts)


def _discover_performance_mean_dfg(log_like) -> Dict[Tuple[str, str], float]:
    # Сначала пытаемся через pm4py PERFORMANCE
    try:
        perf_dfg = dfg_algo.apply(log_like, variant=dfg_algo.Variants.PERFORMANCE)
        # Значения могут быть: число, список, словарь с суммой/средним
        def _mean_from_value(v) -> Optional[float]:
            if v is None:
                return None
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                vals = [float(x) for x in v if x is not None and math.isfinite(float(x))]
                return sum(vals) / len(vals) if vals else None
            if isinstance(v, dict):
                if "mean" in v and v["mean"] is not None:
                    return float(v["mean"]) if math.isfinite(float(v["mean"])) else None
                if "sum" in v and "count" in v and v["count"]:
                    try:
                        mean_val = float(v["sum"]) / float(v["count"])  # type: ignore
                        return mean_val if math.isfinite(mean_val) else None
                    except Exception:
                        return None
            return None

        means: Dict[Tuple[str, str], float] = {}
        for k, v in perf_dfg.items():
            m = _mean_from_value(v)
            if m is not None:
                means[k] = m
        if means:
            return means
        # Если не получилось вытащить средние — упадём в фолбэк
    except Exception:
        pass

    # Фолбэк: считаем среднее время между соседними событиями в пределах одного кейса
    if isinstance(log_like, pd.DataFrame):
        df = log_like
    else:
        # конвертируем лог обратно в DF через "плоское" представление, если возможно
        try:
            from pm4py.objects.conversion.log import converter as log_converter
            df = log_converter.apply(log_like, variant=log_converter.Variants.TO_DATA_FRAME)
        except Exception:
            return {}

    df = df.sort_values(["case:concept:name", "time:timestamp"])  # type: ignore
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)

    # Ожидаем типы: activity в "concept:name", время в "time:timestamp"
    for _, g in df.groupby("case:concept:name"):  # type: ignore
        acts = g["concept:name"].tolist()  # type: ignore
        times = g["time:timestamp"].tolist()  # type: ignore
        for i in range(1, len(acts)):
            a, b = acts[i-1], acts[i]
            dt = (times[i] - times[i-1]).total_seconds() if pd.notnull(times[i]) and pd.notnull(times[i-1]) else None
            if dt is not None and dt >= 0:
                sums[(a, b)] += dt
                counts[(a, b)] += 1

    means: Dict[Tuple[str, str], float] = {}
    for k in sums:
        if counts[k] > 0:
            means[k] = sums[k] / counts[k]
    return means


def _compute_edge_percentiles(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    df = df.sort_values(["case:concept:name", "time:timestamp"])  # type: ignore
    per_edge: Dict[Tuple[str, str], list] = defaultdict(list)
    for _, g in df.groupby("case:concept:name"):  # type: ignore
        acts = g["concept:name"].tolist()  # type: ignore
        times = g["time:timestamp"].tolist()  # type: ignore
        for i in range(1, len(acts)):
            a, b = acts[i-1], acts[i]
            t1, t2 = times[i-1], times[i]
            if pd.notnull(t1) and pd.notnull(t2):
                dt = (t2 - t1).total_seconds()
                if dt >= 0:
                    per_edge[(a, b)].append(float(dt))
    result: Dict[Tuple[str, str], Dict[str, float]] = {}
    for k, arr in per_edge.items():
        if not arr:
            continue
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        p50 = median(arr_sorted)
        idx90 = max(0, int(math.ceil(0.9 * n)) - 1)
        p90 = float(arr_sorted[idx90])
        avg = sum(arr_sorted) / n
        result[k] = {"avg": avg, "p50": p50, "p90": p90}
    return result


def _compute_transitions_and_cases(df: pd.DataFrame):
    """Возвращает список переходов и длительности кейсов.
    Переходы: case_id, src, dst, t_start, t_end, delta_s, channel_from, channel_to, worker_from, worker_to.
    """
    df = df.sort_values(["case:concept:name", "time:timestamp"])  # type: ignore
    transitions = []
    case_durations = {}
    for case_id, g in df.groupby("case:concept:name"):  # type: ignore
        g = g.copy()
        acts = g["concept:name"].tolist()  # type: ignore
        times = g["time:timestamp"].tolist()  # type: ignore
        channels = g[g.columns[g.columns.str.lower().str.contains("канал")].tolist()[0]] if any(g.columns.str.lower().str.contains("канал")) else [None]*len(g)
        workers = g[g.columns[g.columns.str.lower().str.contains("имя работника")].tolist()[0]] if any(g.columns.str.lower().str.contains("имя работника")) else [None]*len(g)
        # Длительность кейса
        if len(times) >= 2 and pd.notnull(times[0]) and pd.notnull(times[-1]):
            case_durations[case_id] = (times[-1] - times[0]).total_seconds()
        # Переходы
        ch = channels if hasattr(channels, "tolist") else channels
        wk = workers if hasattr(workers, "tolist") else workers
        ch_list = ch.tolist() if hasattr(ch, "tolist") else list(ch)
        wk_list = wk.tolist() if hasattr(wk, "tolist") else list(wk)
        for i in range(1, len(acts)):
            t1, t2 = times[i-1], times[i]
            if pd.notnull(t1) and pd.notnull(t2):
                dt = (t2 - t1).total_seconds()
            else:
                dt = None
            transitions.append({
                "case_id": case_id,
                "src": acts[i-1],
                "dst": acts[i],
                "t_start": t1,
                "t_end": t2,
                "delta_s": dt,
                "channel_from": ch_list[i-1] if i-1 < len(ch_list) else None,
                "channel_to": ch_list[i] if i < len(ch_list) else None,
                "worker_from": wk_list[i-1] if i-1 < len(wk_list) else None,
                "worker_to": wk_list[i] if i < len(wk_list) else None,
            })
    return transitions, case_durations


def _export_sla_breaches(transitions_df: pd.DataFrame, sla_map: Dict[Tuple[str, str], float], out_csv: str):
    rows = []
    for _, r in transitions_df.iterrows():
        pair = (str(r["src"]), str(r["dst"]))
        if r["delta_s"] is None:
            continue
        thr = sla_map.get(pair)
        if thr is not None and float(r["delta_s"]) > float(thr):
            rows.append({
                "case_id": r["case_id"], "src": r["src"], "dst": r["dst"],
                "delta_s": float(r["delta_s"]), "sla_s": float(thr),
                "t_start": r["t_start"], "t_end": r["t_end"],
                "channel_from": r.get("channel_from"), "channel_to": r.get("channel_to"),
                "worker_from": r.get("worker_from"), "worker_to": r.get("worker_to"),
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)


def _export_handoff_and_pingpong(transitions_df: pd.DataFrame, out_matrix_csv: str, out_pingpong_csv: str):
    # Матрица передач по сотрудникам
    def normalize(v):
        return str(v) if pd.notnull(v) else "UNKNOWN"
    hand = Counter()
    for _, r in transitions_df.iterrows():
        a = normalize(r.get("worker_from"))
        b = normalize(r.get("worker_to"))
        hand[(a, b)] += 1
    if hand:
        pd.DataFrame([{"from": k[0], "to": k[1], "count": v} for k, v in hand.items()]).to_csv(out_matrix_csv, index=False)
    # Пинг‑понг: A->B затем B->A в рамках кейса
    ping = Counter()
    for case_id, grp in transitions_df.groupby("case_id"):
        grp = grp.sort_values("t_start")
        pairs = list(zip(grp["worker_from"].astype(str), grp["worker_to"].astype(str)))
        for i in range(1, len(pairs)):
            if pairs[i-1][0] == pairs[i][1] and pairs[i-1][1] == pairs[i][0] and pairs[i][0] != pairs[i][1]:
                ping[(pairs[i][0], pairs[i][1])] += 1
    if ping:
        pd.DataFrame([{"a": k[0], "b": k[1], "ping_pongs": v} for k, v in ping.items()]).to_csv(out_pingpong_csv, index=False)


def _export_stage_kpi(transitions_df: pd.DataFrame, out_csv: str):
    by_stage: Dict[str, list] = defaultdict(list)
    total = 0.0
    for _, r in transitions_df.iterrows():
        dt = r.get("delta_s")
        if dt is None:
            continue
        by_stage[str(r["src"])].append(float(dt))
        total += float(dt)
    rows = []
    for stage, arr in by_stage.items():
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        p50 = median(arr_sorted)
        p90 = arr_sorted[max(0, int(math.ceil(0.9 * n)) - 1)]
        share = (sum(arr_sorted) / total) if total > 0 else 0.0
        rows.append({"stage": stage, "count": n, "p50_s": p50, "p90_s": p90, "time_share": share})
    if rows:
        pd.DataFrame(rows).sort_values("p90_s", ascending=False).to_csv(out_csv, index=False)


def _export_variant_explorer(df: pd.DataFrame, out_csv: str):
    df = df.sort_values(["case:concept:name", "time:timestamp"])  # type: ignore
    rows = []
    cur_case = None
    seq = []
    t_first = None
    t_last = None
    channel_col = next((c for c in df.columns if c.lower() == "канал"), None)
    type_col = next((c for c in df.columns if c.lower().startswith("тип страхового случая")), None)
    case_to_channel = {}
    case_to_type = {}
    for _, r in df.iterrows():
        cid = r["case:concept:name"]
        if cur_case is None:
            cur_case = cid
            t_first = r["time:timestamp"]
        if cid != cur_case:
            if seq and pd.notnull(t_first) and pd.notnull(t_last):
                rows.append({
                    "case_id": cur_case,
                    "variant": "→".join(map(str, seq)),
                    "duration_s": (t_last - t_first).total_seconds(),
                    "channel": case_to_channel.get(cur_case),
                    "case_type": case_to_type.get(cur_case)
                })
            cur_case = cid
            seq = []
            t_first = r["time:timestamp"]
        seq.append(r["concept:name"])  # type: ignore
        t_last = r["time:timestamp"]
        if channel_col is not None and cid not in case_to_channel:
            case_to_channel[cid] = r[channel_col]
        if type_col is not None and cid not in case_to_type:
            case_to_type[cid] = r[type_col]
    # last
    if seq and pd.notnull(t_first) and pd.notnull(t_last):
        rows.append({
            "case_id": cur_case,
            "variant": "→".join(map(str, seq)),
            "duration_s": (t_last - t_first).total_seconds(),
            "channel": case_to_channel.get(cur_case),
            "case_type": case_to_type.get(cur_case)
        })
    if not rows:
        return
    dfv = pd.DataFrame(rows)
    agg = dfv.groupby("variant").agg(
        count=("case_id", "nunique"),
        median_s=("duration_s", "median"),
        mean_s=("duration_s", "mean")
    ).reset_index()
    # топ признаки
    top_chan = dfv.groupby(["variant", "channel"])['case_id'].nunique().reset_index().sort_values(['variant','case_id'], ascending=[True,False]).groupby('variant').first().reset_index().rename(columns={'channel':'top_channel','case_id':'cases_channel'})
    top_type = dfv.groupby(["variant", "case_type"])['case_id'].nunique().reset_index().sort_values(['variant','case_id'], ascending=[True,False]).groupby('variant').first().reset_index().rename(columns={'case_type':'top_case_type','case_id':'cases_type'})
    out = agg.merge(top_chan, on='variant', how='left').merge(top_type, on='variant', how='left')
    out.sort_values(['count','median_s'], ascending=[False, True]).to_csv(out_csv, index=False)


def _export_anomalies(df: pd.DataFrame, transitions_df: pd.DataFrame, out_csv: str):
    # Дуринг p99
    case_durations = []
    for cid, g in df.groupby("case:concept:name"):
        g = g.sort_values("time:timestamp")
        if len(g) >= 2:
            t = (g.iloc[-1]["time:timestamp"] - g.iloc[0]["time:timestamp"]).total_seconds()
            case_durations.append((cid, t, len(g)))
    if not case_durations:
        return
    import numpy as np
    arr = np.array([t for _, t, _ in case_durations], dtype=float)
    thr = float(np.percentile(arr, 99))
    # метрики по лупам и пинг‑понгу
    loops_by_case = Counter()
    ping_by_case = Counter()
    for cid, g in transitions_df.groupby("case_id"):
        g = g.sort_values("t_start")
        # loops A->A
        loops_by_case[cid] = int((g["src"] == g["dst"]).sum())
        # ping‑pong
        pairs = list(zip(g["worker_from"].astype(str), g["worker_to"].astype(str)))
        pp = 0
        for i in range(1, len(pairs)):
            if pairs[i-1][0] == pairs[i][1] and pairs[i-1][1] == pairs[i][0] and pairs[i][0] != pairs[i][1]:
                pp += 1
        ping_by_case[cid] = pp
    rows = []
    for cid, dur, n in case_durations:
        if dur >= thr:
            rows.append({"case_id": cid, "duration_s": dur, "events": n, "loops": int(loops_by_case.get(cid, 0)), "ping_pong": int(ping_by_case.get(cid, 0))})
    if rows:
        pd.DataFrame(rows).sort_values("duration_s", ascending=False).to_csv(out_csv, index=False)


def _export_returns_and_starts(df: pd.DataFrame, out_returns: str, out_return_start: str):
    df = df.sort_values(["case:concept:name", "time:timestamp"])  # type: ignore
    returns = Counter()
    ret_to_start = Counter()
    for cid, g in df.groupby("case:concept:name"):  # type: ignore
        acts = g["concept:name"].tolist()  # type: ignore
        if not acts:
            continue
        start = acts[0]
        seen = set([start])
        prev = start
        for i in range(1, len(acts)):
            cur = acts[i]
            if cur == start and cur != prev:
                ret_to_start[(prev, cur)] += 1
            if cur in seen and cur != prev:
                returns[(prev, cur)] += 1
            seen.add(cur)
            prev = cur
    if returns:
        pd.DataFrame([{"src": a, "dst": b, "count": c} for (a, b), c in sorted(returns.items(), key=lambda kv: -kv[1])]).to_csv(out_returns, index=False)
    if ret_to_start:
        pd.DataFrame([{"src": a, "dst": b, "count": c} for (a, b), c in sorted(ret_to_start.items(), key=lambda kv: -kv[1])]).to_csv(out_return_start, index=False)


def _export_activity_ping_pong(df: pd.DataFrame, out_csv: str):
    df = df.sort_values(["case:concept:name", "time:timestamp"])  # type: ignore
    ping = Counter()
    for cid, g in df.groupby("case:concept:name"):  # type: ignore
        acts = g["concept:name"].tolist()  # type: ignore
        for i in range(2, len(acts)):
            if acts[i] == acts[i-2] and acts[i] != acts[i-1]:
                a, b = acts[i-2], acts[i-1]
                if a != b:
                    ping[(a, b)] += 1
    if ping:
        pd.DataFrame([{"a": a, "b": b, "count": c} for (a, b), c in sorted(ping.items(), key=lambda kv: -kv[1])]).to_csv(out_csv, index=False)


def _export_rare_activities(df: pd.DataFrame, out_csv: str, rare_threshold: float = 0.02):
    total_cases = df["case:concept:name"].nunique()  # type: ignore
    per_act = df.groupby("concept:name")["case:concept:name"].nunique().reset_index().rename(columns={"concept:name": "activity", "case:concept:name": "cases"})  # type: ignore
    per_act["share"] = per_act["cases"] / float(total_cases) if total_cases else 0.0
    rare = per_act.sort_values("share").query("share < @rare_threshold")
    if not rare.empty:
        rare.to_csv(out_csv, index=False)


def _export_manual_steps(df: pd.DataFrame, out_csv: str):
    # Ищем признаки ручных шагов: наличие исполнителя и тип канала
    channel_col = next((c for c in df.columns if c.lower() == "канал"), None)
    worker_col = next((c for c in df.columns if c.lower() == "имя работника"), None)
    df2 = df.copy()
    if worker_col is not None:
        df2["worker_missing"] = df2[worker_col].isna() | (df2[worker_col].astype(str).str.strip() == "")
    else:
        df2["worker_missing"] = True
    manual_channels = {"Звонок", "Телефон", "Чат", "Email", "Почта", "Личное посещение"}
    if channel_col is not None:
        df2["manual_channel"] = df2[channel_col].astype(str).isin(manual_channels)
    else:
        df2["manual_channel"] = False
    agg = df2.groupby("concept:name").agg(
        events=("concept:name", "count"),
        worker_missing_events=("worker_missing", "sum"),
        manual_channel_events=("manual_channel", "sum")
    ).reset_index().rename(columns={"concept:name": "activity"})
    agg.to_csv(out_csv, index=False)


def _export_unsuccess_outcomes(df: pd.DataFrame, out_csv: str):
    df = df.sort_values(["case:concept:name", "time:timestamp"])  # type: ignore
    rows = []
    last_by_case = df.groupby("case:concept:name").tail(1)  # type: ignore
    # Опознаём «Отклонение претензии» как неуспешный исход
    last_by_case = last_by_case.copy()
    last_by_case["unsuccess"] = last_by_case["concept:name"].astype(str).str.contains("Отклонение претензии", na=False)  # type: ignore
    # Длительности кейсов
    durations = {}
    for cid, g in df.groupby("case:concept:name"):  # type: ignore
        if len(g) >= 2:
            durations[cid] = (g.iloc[-1]["time:timestamp"] - g.iloc[0]["time:timestamp"]).total_seconds()
    chan_col = next((c for c in df.columns if c.lower() == "канал"), None)
    type_col = next((c for c in df.columns if c.lower().startswith("тип страхового случая")), None)
    for _, r in last_by_case.iterrows():
        cid = r["case:concept:name"]
        rows.append({
            "case_id": cid,
            "unsuccess": bool(r["unsuccess"]),
            "duration_s": durations.get(cid),
            "channel": r.get(chan_col) if chan_col else None,
            "case_type": r.get(type_col) if type_col else None,
            "last_activity": r["concept:name"],
        })
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)


def _export_cohort_trends(df: pd.DataFrame, out_csv: str):
    df = df.copy()
    df["month"] = pd.to_datetime(df["time:timestamp"]).dt.to_period("M").dt.to_timestamp()  # type: ignore
    chan_col = next((c for c in df.columns if c.lower() == "канал"), None)
    type_col = next((c for c in df.columns if c.lower().startswith("тип страхового случая")), None)
    # длительность кейса по month и сегментам: считаем по последнему событию кейса в месяце закрытия
    rows = []
    for (cid, m), g in df.groupby(["case:concept:name", "month"]):  # type: ignore
        g = g.sort_values("time:timestamp")
        if len(g) >= 2:
            dur = (g.iloc[-1]["time:timestamp"] - g.iloc[0]["time:timestamp"]).total_seconds()
            rows.append({
                "case_id": cid,
                "month": g.iloc[-1]["month"],
                "duration_s": dur,
                "channel": g.iloc[-1][chan_col] if chan_col else None,
                "case_type": g.iloc[-1][type_col] if type_col else None,
            })
    if not rows:
        return
    cdf = pd.DataFrame(rows)
    out_rows = []
    for seg_col in [None, "channel", "case_type"]:
        grp_cols = ["month"] + ([seg_col] if seg_col else [])
        tmp = cdf.groupby(grp_cols)["duration_s"].agg(["count", "median", lambda x: np.percentile(x, 90)]).reset_index()
        tmp = tmp.rename(columns={"median": "median_s", "<lambda_0>": "p90_s"})
        tmp["segment"] = seg_col or "all"
        out_rows.append(tmp)
    out_df = pd.concat(out_rows, ignore_index=True)
    out_df.to_csv(out_csv, index=False)


def _export_duration_trend_test(df: pd.DataFrame, out_txt: str):
    # Тест на монотонный тренд медианы по месяцам (Кендалл)
    df = df.copy()
    df["month"] = pd.to_datetime(df["time:timestamp"]).dt.to_period("M").dt.to_timestamp()  # type: ignore
    # строим медиану длительности кейсов по месяцам
    rows = []
    for cid, g in df.groupby("case:concept:name"):  # type: ignore
        g = g.sort_values("time:timestamp")
        if len(g) >= 2:
            rows.append({
                "case_id": cid,
                "month": g.iloc[-1]["month"],
                "duration_s": (g.iloc[-1]["time:timestamp"] - g.iloc[0]["time:timestamp"]).total_seconds()
            })
    if not rows:
        return
    cdf = pd.DataFrame(rows)
    series = cdf.groupby("month")["duration_s"].median().reset_index().sort_values("month")
    x = np.arange(len(series))
    y = series["duration_s"].values
    tau, p = kendalltau(x, y)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Kendall tau: {tau:.4f}, p-value: {p:.6f}\n")
        f.write("increasing\n" if tau > 0 and p < 0.05 else ("decreasing\n" if tau < 0 and p < 0.05 else "no_trend\n"))


def build_and_save_dfg(
    input_csv: str,
    output_png: str,
    case_col: str,
    activity_col: str,
    ts_col: str,
    ts_format: Optional[str],
    sep: str,
    encoding: str,
    min_freq: int,
    rankdir: str,
    sla_csv: Optional[str] = None,
    top_variant_csv: Optional[str] = None,
    rare_edge_threshold: int = 5,
    bottleneck_p90_threshold_s: Optional[float] = None,
    export_reports: bool = True,
    export_svg: bool = True,
    export_html: bool = True,
    tables_dir: str = "tables",
    rare_activity_threshold: float = 0.02
) -> None:
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Не найден входной CSV: {input_csv}")

    df = pd.read_csv(input_csv, sep=sep, encoding=encoding)
    formatted_df = _format_dataframe(df, case_col, activity_col, ts_col, ts_format)
    log_like = _to_event_log(formatted_df)

    freq_dfg = _discover_frequency_dfg(log_like)
    perf_mean_dfg = _discover_performance_mean_dfg(log_like)
    edge_stats = _compute_edge_percentiles(formatted_df)

    # SLA карта
    sla_map: Dict[Tuple[str, str], float] = {}
    if sla_csv and os.path.isfile(sla_csv):
        try:
            sla_df = pd.read_csv(sla_csv)
            for _, r in sla_df.iterrows():
                src = str(r["src"]) if "src" in sla_df.columns else str(r[0])
                dst = str(r["dst"]) if "dst" in sla_df.columns else str(r[1])
                p90_s = float(r["p90_s"]) if "p90_s" in sla_df.columns else float(r[3])
                sla_map[(src, dst)] = p90_s
        except Exception:
            pass

    # Эталонные рёбра из топ-варианта
    golden_edges: set = set()
    if top_variant_csv and os.path.isfile(top_variant_csv):
        try:
            vt = pd.read_csv(top_variant_csv)
            vt = vt.sort_values("count", ascending=False)
            if len(vt) > 0:
                seq_str = vt.iloc[0]["variant"]
                seq = [s.strip() for s in str(seq_str).split("→") if s.strip()]
                for i in range(1, len(seq)):
                    golden_edges.add((seq[i-1], seq[i]))
        except Exception:
            pass

    # Узлы
    activities = sorted(set(formatted_df["concept:name"].unique()))  # type: ignore

    dot = Digraph("DFG", format="png")
    dot.attr(rankdir=rankdir)
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#f8f9fb", color="#b5bdd6", fontname="Arial", fontsize="10")

    # Масштаб по частоте и легенда
    max_freq = max(freq_dfg.values()) if freq_dfg else 1
    def edge_width(f: int) -> str:
        if max_freq <= 1:
            return "1.0"
        w = 0.5 + 3.5 * (f / max_freq)
        return f"{w:.2f}"
    with dot.subgraph(name="cluster_legend") as lg:
        lg.attr(label="Легенда", color="#b5bdd6", fontname="Arial", fontsize="10")
        lg.node("leg1", label="label = частота | ср.время (avg)", shape="note", fillcolor="#eef2ff")
        lg.node("leg2", label="красный = SLA p90 превышен", shape="note", fillcolor="#ffe7e7")
        lg.node("leg3", label="оранжевый = высокий p90", shape="note", fillcolor="#fff4e5")
        lg.node("leg4", label="серый пунктир = редкие рёбра", shape="note", fillcolor="#f3f4f6")
        lg.node("leg5", label="фиолетовый = самопетля", shape="note", fillcolor="#f5e6ff")
        lg.node("leg6", label="синий пунктир = вне основной траектории", shape="note", fillcolor="#e7f0ff")

    for act in activities:
        dot.node(str(act), label=str(act))

    for (a, b), f in sorted(freq_dfg.items(), key=lambda kv: (-kv[1], kv[0])):
        if f < min_freq:
            continue
        mean_sec = perf_mean_dfg.get((a, b))
        mean_h = _humanize_seconds(mean_sec)
        label = f"{f} | {mean_h}"
        color = "#7c88a6"
        penwidth = edge_width(f)
        style = "solid"
        if f <= rare_edge_threshold:
            color = "#9ca3af"; style = "dashed"
        if a == b:
            color = "#7c3aed"
        p90 = edge_stats.get((a, b), {}).get("p90")
        if bottleneck_p90_threshold_s is not None and p90 is not None and p90 >= bottleneck_p90_threshold_s:
            color = "#f59e0b"
        sla = sla_map.get((a, b))
        if sla is not None and p90 is not None and p90 > sla:
            color = "#ef4444"
        if golden_edges and (a, b) not in golden_edges:
            if style == "solid":
                style = "dashed"
            color = "#2563eb"
        dot.edge(str(a), str(b), label=label, color=color, penwidth=penwidth, style=style)

    # Сохранение графа (PNG + опционально SVG/HTML)
    out_dir = os.path.dirname(output_png)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    base_no_ext = os.path.splitext(output_png)[0]
    dot.render(filename=base_no_ext, cleanup=True)
    if export_svg:
        dot.format = "svg"
        dot.render(filename=base_no_ext, cleanup=True)
        dot.format = "png"
    if export_html:
        svg_path = f"{base_no_ext}.svg"
        if os.path.isfile(svg_path):
            html_path = f"{base_no_ext}.html"
            with open(svg_path, "r", encoding="utf-8") as f:
                svg = f.read()
            html = f"""<!doctype html><html><head><meta charset='utf-8'><title>DFG</title>
<style>body{{font-family:Arial,sans-serif;background:#0b1020;color:#e8ecf6;margin:0;padding:16px}} .wrap{{background:#fff;border-radius:8px;padding:12px}} .legend{{margin-bottom:12px}}</style>
</head><body><div class='wrap'>{svg}</div></body></html>"""
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

    # Доп. отчёты
    if export_reports:
        os.makedirs(tables_dir, exist_ok=True)
        trans_rows, case_durations = _compute_transitions_and_cases(formatted_df)
        trans_df = pd.DataFrame(trans_rows)
        # SLA breaches per case
        if sla_map:
            _export_sla_breaches(trans_df, sla_map, os.path.join(tables_dir, "sla_breaches_cases.csv"))
        # Handoff & ping-pong
        _export_handoff_and_pingpong(trans_df, os.path.join(tables_dir, "handoff_matrix.csv"), os.path.join(tables_dir, "ping_pong_pairs.csv"))
        # Stage KPI
        _export_stage_kpi(trans_df, os.path.join(tables_dir, "stage_kpi.csv"))
        # Variant explorer
        _export_variant_explorer(formatted_df, os.path.join(tables_dir, "variant_explorer.csv"))
        # Anomalies
        _export_anomalies(formatted_df, trans_df, os.path.join(tables_dir, "anomalies_cases.csv"))
        # Returns and starts
        _export_returns_and_starts(formatted_df, os.path.join(tables_dir, "returns.csv"), os.path.join(tables_dir, "return_to_start.csv"))
        # Activity ping-pong
        _export_activity_ping_pong(formatted_df, os.path.join(tables_dir, "activity_ping_pong.csv"))
        # Rare activities
        _export_rare_activities(formatted_df, os.path.join(tables_dir, "rare_activities.csv"), rare_activity_threshold)
        # Manual steps
        _export_manual_steps(formatted_df, os.path.join(tables_dir, "manual_steps.csv"))
        # Unsuccessful outcomes
        _export_unsuccess_outcomes(formatted_df, os.path.join(tables_dir, "unsuccess_outcomes.csv"))
        # Cohort trends and trend test
        _export_cohort_trends(formatted_df, os.path.join(tables_dir, "cohort_trends.csv"))
        _export_duration_trend_test(formatted_df, os.path.join(tables_dir, "duration_trend_test.txt"))


# -------------------------------
# CLI
# -------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Построение DFG графа с частотами и средним временем переходов (pm4py + Graphviz)"
    )
    parser.add_argument("-i", "--input", required=True, help="Путь к входному CSV")
    parser.add_argument("-o", "--output", default="dfg_combined.png", help="Выходной PNG (будет перезаписан)")
    parser.add_argument("--case-col", default="case_id", help="Колонка идентификатора кейса")
    parser.add_argument("--activity-col", default="activity", help="Колонка названия активности")
    parser.add_argument("--timestamp-col", default="timestamp", help="Колонка метки времени")
    parser.add_argument("--timestamp-format", default=None, help="Формат времени для strptime, если не ISO (например, %%Y-%%m-%%d %%H:%%M:%%S)")
    parser.add_argument("--sep", default=",", help="Разделитель CSV (по умолчанию ',')")
    parser.add_argument("--encoding", default="utf-8", help="Кодировка CSV")
    parser.add_argument("--min-freq", type=int, default=1, help="Минимальная частота ребра для отображения")
    parser.add_argument("--rankdir", choices=["LR", "TB", "BT", "RL"], default="TB", help="Направление графа (LR/TB/BT/RL)")
    parser.add_argument("--sla-csv", default="tables/edges_sla_template.csv", help="CSV с колонками src,dst,p90_s для SLA")
    parser.add_argument("--top-variant-csv", default="tables/variants_top.csv", help="CSV с колонками variant,count для эталонного пути")
    parser.add_argument("--rare-edge-threshold", type=int, default=5, help="Порог частоты для редких рёбер")
    parser.add_argument("--bottleneck-p90-threshold-s", type=float, default=43200.0, help="Порог p90 (сек) для подсветки bottleneck")
    parser.add_argument("--no-reports", action="store_true", help="Не генерировать дополнительные отчёты")
    parser.add_argument("--no-svg", action="store_true", help="Не сохранять SVG")
    parser.add_argument("--no-html", action="store_true", help="Не сохранять HTML")
    parser.add_argument("--tables-dir", default="tables", help="Папка для CSV отчётов")
    parser.add_argument("--rare-activity-threshold", type=float, default=0.02, help="Порог доли кейсов для редких активностей")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build_and_save_dfg(
        input_csv=args.input,
        output_png=args.output,
        case_col=args.case_col,
        activity_col=args.activity_col,
        ts_col=args.timestamp_col,
        ts_format=args.timestamp_format,
        sep=args.sep,
        encoding=args.encoding,
        min_freq=args.min_freq,
        rankdir=args.rankdir,
        sla_csv=args.sla_csv,
        top_variant_csv=args.top_variant_csv,
        rare_edge_threshold=args.rare_edge_threshold,
        bottleneck_p90_threshold_s=args.bottleneck_p90_threshold_s,
        export_reports=not args.no_reports,
        export_svg=not args.no_svg,
        export_html=not args.no_html,
        tables_dir=args.tables_dir,
        rare_activity_threshold=args.rare_activity_threshold,
    )
    print(f"Готово. PNG: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
