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
    bottleneck_p90_threshold_s: Optional[float] = None
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

    # Эталонные рёбра из топ-варианта: распарсим первую строку с наибольшим count
    golden_edges: set = set()
    if top_variant_csv and os.path.isfile(top_variant_csv):
        try:
            vt = pd.read_csv(top_variant_csv)
            vt = vt.sort_values("count", ascending=False)
            if len(vt) > 0:
                seq_str = vt.iloc[0]["variant"]
                # строка вида: "('A', 'B', 'C')"
                seq = [s.strip().strip("'\"") for s in str(seq_str).strip("() ").split(",") if s.strip()]
                seq = [s for s in seq if s not in ["", "'"]]
                # восстановим пары
                clean = []
                # аккуратно собрать с учётом запятых в названиях не требуется, т.к. в примере их нет
                for token in seq:
                    if token.startswith("'") and token.endswith("'"):
                        clean.append(token[1:-1])
                    else:
                        clean.append(token)
                for i in range(1, len(clean)):
                    golden_edges.add((clean[i-1], clean[i]))
        except Exception:
            pass

    # Узлы
    activities = sorted(set(formatted_df["concept:name"].unique()))  # type: ignore

    dot = Digraph("DFG", format="png")
    dot.attr(rankdir=rankdir)
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#f8f9fb", color="#b5bdd6", fontname="Arial", fontsize="10")

    # Рисуем рёбра со стилями
    # Подготовка масштаба толщины по частоте
    max_freq = max(freq_dfg.values()) if freq_dfg else 1
    def edge_width(f: int) -> str:
        # от 0.5 до 4.0
        if max_freq <= 1:
            return "1.0"
        w = 0.5 + 3.5 * (f / max_freq)
        return f"{w:.2f}"

    # Легенда
    with dot.subgraph(name="cluster_legend") as lg:
        lg.attr(label="Легенда", color="#b5bdd6", fontname="Arial", fontsize="10")
        lg.node("leg1", label="label = частота | ср.время (avg)", shape="note", fillcolor="#eef2ff")
        lg.node("leg2", label="красный = SLA p90 превышен", shape="note", fillcolor="#ffe7e7")
        lg.node("leg3", label="оранжевый = высокий p90", shape="note", fillcolor="#fff4e5")
        lg.node("leg4", label="серый пунктир = редкие рёбра", shape="note", fillcolor="#f3f4f6")
        lg.node("leg5", label="фиолетовый = самопетля", shape="note", fillcolor="#f5e6ff")
        lg.node("leg6", label="синий пунктир = вне основной траектории", shape="note", fillcolor="#e7f0ff")

    # Узлы
    for act in activities:
        dot.node(str(act), label=str(act))

    # Рёбра
    for (a, b), f in sorted(freq_dfg.items(), key=lambda kv: (-kv[1], kv[0])):
        if f < min_freq:
            continue
        mean_sec = perf_mean_dfg.get((a, b))
        mean_h = _humanize_seconds(mean_sec)
        label = f"{f} | {mean_h}"

        # стиль
        color = "#7c88a6"
        penwidth = edge_width(f)
        style = "solid"

        # редкие рёбра
        if f <= rare_edge_threshold:
            color = "#9ca3af"
            style = "dashed"

        # самопетли
        if a == b:
            color = "#7c3aed"  # фиолетовый

        # высокий p90
        p90 = edge_stats.get((a, b), {}).get("p90")
        if bottleneck_p90_threshold_s is not None and p90 is not None and p90 >= bottleneck_p90_threshold_s:
            color = "#f59e0b"  # оранжевый

        # SLA превышение
        sla = sla_map.get((a, b))
        if sla is not None and p90 is not None and p90 > sla:
            color = "#ef4444"  # красный

        # вне основной траектории
        if golden_edges and (a, b) not in golden_edges:
            if style == "solid":
                style = "dashed"
            color = "#2563eb"

        dot.edge(str(a), str(b), label=label, color=color, penwidth=penwidth, style=style)

    # Сохранение
    out_dir = os.path.dirname(output_png)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    dot.render(filename=os.path.splitext(output_png)[0], cleanup=True)


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
    )
    print(f"Готово. PNG: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
