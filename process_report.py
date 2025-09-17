# -*- coding: utf-8 -*-
# process_report.py — генерит картинки и CSV для поиска узких мест

import argparse
import math
import os
import sys
from typing import Dict, Tuple, Optional
from collections import Counter
import glob
import difflib

import pandas as pd
import pm4py
from pm4py.algo.discovery.dfg import algorithm as dfg_algo
from pm4py.statistics.traces.generic.log import case_statistics as case_stats
from graphviz import Digraph


import os
import glob
import sys
import shutil

# 📂 Рабочая директория
WORK_DIR = "/Users/anna/Documents/Кейс чемпионат сбер/PM_CASE_CH/"
TABLES_DIR = os.path.join(WORK_DIR, "tables")

# ❌ Не трогаем эти файлы
KEEP = {
    "case_championship_last.csv",
    os.path.basename(sys.argv[0]),  # сам .py
    "README.md",
    ".gitignore",
}

def clean_directory():
    for path in glob.glob(os.path.join(WORK_DIR, "*")):
        fname = os.path.basename(path)
        if fname not in KEEP:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"Удалён файл: {fname}")
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Удалена папка: {fname}")
            except Exception as e:
                print(f"⚠️ Ошибка при удалении {fname}: {e}")

# 1️⃣ Сначала чистим
clean_directory()

# 2️⃣ Создаём папку для таблиц заново
os.makedirs(TABLES_DIR, exist_ok=True)

# 3️⃣ Меняем текущую директорию процесса
os.chdir(WORK_DIR)


# Создаем папку для таблиц, если её нет
TABLES_DIR = "tables"
if not os.path.exists(TABLES_DIR):
    os.makedirs(TABLES_DIR)


def resolve_csv_path(path: str) -> str:
    """
    Пытаемся корректно найти CSV по указанному пути:
    - если существует как указан (абсолютный/относительный) — берём его;
    - если нет — ищем в директории скрипта;
    - затем — ищем по похожим именам (case-insensitive) в cwd и в папке скрипта (рекурсивно);
    - если найдено несколько — берём первый попавшийся;
    - если ничего не найдено — выбрасываем информативную ошибку со списком найденных CSV.
    """
    # 1) прямой путь
    if os.path.isabs(path) and os.path.exists(path):
        return os.path.abspath(path)
    if os.path.exists(path):
        return os.path.abspath(path)

    # 2) путь относительно директории скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidate = os.path.join(script_dir, path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # 3) expanduser
    candidate = os.path.expanduser(path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # 4) поиск в cwd и script_dir по похожему имени (case-insensitive)
    name = os.path.basename(path).lower()
    for d in (os.getcwd(), script_dir):
        try:
            for f in os.listdir(d):
                if f.lower() == name or name in f.lower():
                    p = os.path.join(d, f)
                    if os.path.isfile(p):
                        return os.path.abspath(p)
        except Exception:
            pass

    # 5) рекурсивный поиск в папке скрипта (ограничение глубины можно добавить при необходимости)
    for root, _, files in os.walk(script_dir):
        for f in files:
            if f.lower() == name or name in f.lower():
                return os.path.abspath(os.path.join(root, f))

    # 6) если ничего не найдено — собрать список доступных CSV для диагностики и бросить ошибку
    found = []
    for d in (os.getcwd(), script_dir):
        try:
            for f in os.listdir(d):
                if f.lower().endswith(".csv"):
                    found.append(os.path.abspath(os.path.join(d, f)))
        except Exception:
            pass

    raise FileNotFoundError(
        f"CSV not found: {path}\n"
        f"Searched locations:\n  cwd = {os.getcwd()}\n  script_dir = {script_dir}\n"
        f"Found CSV files (in those dirs): {found}\n"
        f"Run the script with --csv /full/path/to/file.csv if needed."
    )


# ---------- Чтение и подготовка лога ----------
def build_event_log(
    csv_path: str,
    encoding: str = "utf-8-sig",
    tz_local: Optional[str] = None,
    id_col: str = "ID",
    act_col: str = "Событие",
    ts_col: str = "Время",
) -> pm4py.objects.log.obj.EventLog:
    # csv_path уже должен быть проверен/разрешён через resolve_csv_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found (after resolution): {csv_path}")
    df = pd.read_csv(csv_path, encoding=encoding)
    need = {id_col, act_col, ts_col}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Нет столбцов: {miss}. Нашёл: {list(df.columns)}")
    df = df.rename(columns={id_col: "case:concept:name", act_col: "concept:name", ts_col: "time:timestamp"})
    ts = pd.to_datetime(df["time:timestamp"], errors="coerce")  # без utc=True!
    # tz-логика
    if tz_local:
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tz_local, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
    else:
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
    df["time:timestamp"] = ts  # <-- ВАЖНО: присвоить обратно!
    # чистка и стабильная сортировка
    df = df.dropna(subset=["case:concept:name", "concept:name", "time:timestamp"]).copy()
    df["_row_ix"] = range(len(df))
    df = df.sort_values(["case:concept:name", "time:timestamp", "_row_ix"]).reset_index(drop=True).drop(columns=["_row_ix"])
    ev_df = pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
    return pm4py.convert_to_event_log(ev_df)


# ---------- Подсчёты ----------
def edge_durations_seconds(log):
    out = []
    for trace in log:
        for i in range(len(trace) - 1):
            a = trace[i]["concept:name"]; b = trace[i + 1]["concept:name"]
            t0 = trace[i]["time:timestamp"]; t1 = trace[i + 1]["time:timestamp"]
            if pd.notna(t0) and pd.notna(t1):
                sec = (pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds()
                if sec >= 0:
                    out.append((a, b, sec))
    return out


def path_signature(trace):
    return "→".join([e["concept:name"] for e in trace])


def _scale_penwidth(value: float, vmin: float, vmax: float, min_w=0.7, max_w=6.0) -> float:
    if vmax <= vmin:
        return (min_w + max_w) / 2.0
    x = (value - vmin) / (vmax - vmin)
    x = math.sqrt(max(0.0, min(1.0, x)))
    return min_w + x * (max_w - min_w)


def _sanitize_id(name: str) -> str:
    if name is None:
        name = "unnamed"
    return str(name).replace("\n", " ").replace('"', "'")


# ---------- Визуализации ----------
# ---------- Комбинированная визуализация ----------
def render_dfg_combined(
    dfg_freq: Dict[Tuple[str, str], int],
    dfg_perf: Dict[Tuple[str, str], float],
    out_basename="dfg_combined",
    min_ratio=0.01,
    max_edges=None,
    unit="h"
):
    if not dfg_freq or not dfg_perf:
        print("DFG пуст."); return

    total = sum(dfg_freq.values())
    items = [((a, b), c) for (a, b), c in dfg_freq.items() if c / total >= min_ratio]
    items.sort(key=lambda kv: kv[1], reverse=True)
    if max_edges:
        items = items[:max_edges]

    nodes = set()
    for (a, b), _ in items:
        nodes.update([a, b])

    counts = [c for _, c in items]
    cmin, cmax = min(counts), max(counts)

    def conv(sec):
        if sec is None:
            return None
        if unit == "s":
            return sec
        if unit == "m":
            return sec / 60.0
        if unit == "h":
            return sec / 3600.0
        if unit == "d":
            return sec / 86400.0
        return sec

    # собираем перформансы в выбранных единицах
    perfs_raw = [dfg_perf.get((a, b), None) for (a, b), _ in items]
    perfs = [conv(v) for v in perfs_raw if v is not None]
    if perfs:
        pmin, pmax = min(perfs), max(perfs)
    else:
        pmin, pmax = 0.0, 0.0

    #  функция выбора цвета ребра по SLA-логике (три зоны)
    def edge_color(sec_conv, pmin, pmax):
        if sec_conv is None:
            return "black"
        # если диапазон нулевой — всё в средней/зелёной зоне (не будем делить на ноль)
        if pmax <= pmin:
            return "green"
        thr1 = pmin + (pmax - pmin) * 0.33
        thr2 = pmin + (pmax - pmin) * 0.66
        if sec_conv <= thr1:
            return "green"
        elif sec_conv <= thr2:
            return "yellow"
        else:
            return "red"

    g = Digraph(engine="dot")
    g.attr('graph',
           rankdir='TB',
           size="10,10!",
           ratio="fill",
           splines='spline',
           fontname='DejaVu Sans')
    g.attr("node", shape="box", style="rounded,filled", fontsize="11", fontname='DejaVu Sans')
    g.attr("edge", fontsize="9", fontname='DejaVu Sans')

    # частота узлов
    node_freq = Counter()
    for (a, b), c in items:
        node_freq[a] += c
        node_freq[b] += c
    nfmin, nfmax = min(node_freq.values()), max(node_freq.values())

    # Узлы серые
    for n in sorted(nodes):
        nid = _sanitize_id(n)
        g.node(nid, label=nid, fillcolor="lightgray")

    # Рёбра с трёхцветной логикой
    for (a, b), c in items:
        aw = _sanitize_id(a)
        bw = _sanitize_id(b)
        sec_raw = dfg_perf.get((a, b))
        sec_conv = conv(sec_raw)
        label = f"{c}×" + (f" | {sec_conv:.2f} {unit}" if sec_conv is not None else "")
        color = edge_color(sec_conv, pmin, pmax)

        g.edge(
            aw, bw,
            label=label,
            penwidth=str(round(_scale_penwidth(c, cmin, cmax), 2)),
            color=color,
            fontcolor="gray20"
        )

    g.render(out_basename, format="png", cleanup=True)
    print(f"Сохранено: {out_basename}.png (зелёный/жёлтый/красный)")


# ---------- SLA (опционально) ----------
def parse_sla(s: Optional[str]):
    if not s:
        return {}
    rules = {}
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        left, right = part.split("=")
        src, dst = left.split("->")
        rules[(src.strip(), dst.strip())] = float(right)
    return rules


def sla_breaches(df_edges, rules: Dict[Tuple[str, str], float]):
    rows = []
    for (src, dst), limit in rules.items():
        dfp = df_edges[(df_edges["src"] == src) & (df_edges["dst"] == dst)]
        if len(dfp) == 0:
            rows.append({"src": src, "dst": dst, "count": 0, "breaches": 0, "breach_rate": None, "p90_s": None})
            continue
        count = len(dfp)
        breaches = int((dfp["sec"] > limit).sum())
        breach_rate = breaches / count if count else None
        p90 = float(dfp["sec"].quantile(0.9)) if count else None
        rows.append({"src": src, "dst": dst, "count": count, "breaches": breaches, "breach_rate": breach_rate, "p90_s": p90})
    return pd.DataFrame(rows)


# ---------- Основной ход ----------
def main():
    ap = argparse.ArgumentParser(description="Генератор отчётов по процессу")
    ap.add_argument("--csv", default="case_championship_last.csv", help="Путь к CSV")
    ap.add_argument("--encoding", default="utf-8-sig")
    ap.add_argument("--tz", default=None, help="Напр., Europe/Moscow")
    ap.add_argument("--min_ratio", type=float, default=0.02, help="Фильтр рёбер по доле переходов")
    ap.add_argument("--max_edges", type=int, default=None, help="Топ-ребёр для отрисовки")
    ap.add_argument("--perf_unit", default="h", choices=["s", "m", "h", "d"])
    ap.add_argument("--sla", default=None, help='Формат: "A->B=86400;X->Y=7200" (секунды)')
    args = ap.parse_args()

    try:
        csv_path = resolve_csv_path(args.csv)
        print(f"Использую CSV: {csv_path}")
        log = build_event_log(csv_path, encoding=args.encoding, tz_local=args.tz)
    except Exception as e:
        print(f"Ошибка лога: {e}", file=sys.stderr)
        sys.exit(1)

    # Картинки DFG
    dfg_freq = dfg_algo.apply(log, variant=dfg_algo.Variants.FREQUENCY)
    dfg_perf = dfg_algo.apply(log, variant=dfg_algo.Variants.PERFORMANCE)

    render_dfg_combined(
        dfg_freq,
        dfg_perf,
        out_basename="dfg_combined",
        min_ratio=args.min_ratio,
        max_edges=args.max_edges,
        unit=args.perf_unit)

    # Таблица по рёбрам (времена)
    rows = edge_durations_seconds(log)
    df_edges = pd.DataFrame(rows, columns=["src", "dst", "sec"])
    agg = df_edges.groupby(["src", "dst"])["sec"].agg(
        count="count",
        avg_s="mean",
        p50_s=lambda s: float(pd.Series(s).quantile(0.5)),
        p90_s=lambda s: float(pd.Series(s).quantile(0.9)),
        max_s="max"
    ).reset_index().sort_values("p90_s", ascending=False)
    agg.to_csv(os.path.join(TABLES_DIR, "bottlenecks_edges.csv"), index=False, encoding="utf-8-sig")
    print(f"Сохранено: {os.path.join(TABLES_DIR, 'bottlenecks_edges.csv')}")

    total_transitions = agg["count"].sum()
    agg["ratio"] = agg["count"] / (total_transitions if total_transitions else 1)
    rare_costly = agg[(agg["ratio"] < 0.02) & (agg["p90_s"] > 3600)]
    rare_costly.to_csv(os.path.join(TABLES_DIR, "rare_costly_edges.csv"), index=False, encoding="utf-8-sig")
    print(f"Сохранено: {os.path.join(TABLES_DIR, 'rare_costly_edges.csv')}")

    # Длительность кейсов
    durations = case_stats.get_all_case_durations(log)  # сек
    df_cases = pd.DataFrame({"case_duration_s": durations})
    df_cases["case_duration_h"] = df_cases["case_duration_s"] / 3600.0
    df_cases.to_csv(os.path.join(TABLES_DIR, "cases_durations.csv"), index=False, encoding="utf-8-sig")
    print(f"Сохранено: {os.path.join(TABLES_DIR, 'cases_durations.csv')}")

    # Варианты
    variants = case_stats.get_variant_statistics(log)
    df_variants = pd.DataFrame(variants).sort_values("count", ascending=False)
    df_variants.to_csv(os.path.join(TABLES_DIR, "variants_top.csv"), index=False, encoding="utf-8-sig")
    print(f"Сохранено: {os.path.join(TABLES_DIR, 'variants_top.csv')}")

    # быстрые/медленные пути
    q90 = df_cases["case_duration_s"].quantile(0.9) if len(df_cases) else 0
    q50 = df_cases["case_duration_s"].quantile(0.5) if len(df_cases) else 0
    slow_cases, fast_cases = set(), set()
    for trace in log:
        cid = trace.attributes.get("concept:name")
        ts = [e["time:timestamp"] for e in trace]
        dur = (max(ts) - min(ts)).total_seconds() if len(ts) >= 2 else 0
        if dur >= q90:
            slow_cases.add(cid)
        elif dur <= q50:
            fast_cases.add(cid)
    slow_paths, fast_paths = Counter(), Counter()
    for trace in log:
        cid = trace.attributes.get("concept:name")
        sig = path_signature(trace)
        if cid in slow_cases:
            slow_paths[sig] += 1
        elif cid in fast_cases:
            fast_paths[sig] += 1
    pd.DataFrame(slow_paths.items(), columns=["variant", "count"]).sort_values("count", ascending=False) \
        .to_csv(os.path.join(TABLES_DIR, "variants_slow_top.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fast_paths.items(), columns=["variant", "count"]).sort_values("count", ascending=False) \
        .to_csv(os.path.join(TABLES_DIR, "variants_fast_top.csv"), index=False, encoding="utf-8-sig")
    print(f"Сохранено: {os.path.join(TABLES_DIR, 'variants_slow_top.csv')}, {os.path.join(TABLES_DIR, 'variants_fast_top.csv')}")

    # Rework / повторы
    def count_adjacent_repeats(trace):
        names = [e["concept:name"] for e in trace]
        return sum(1 for i in range(1, len(names)) if names[i] == names[i - 1])

    def count_return_loops(trace):
        names = [e["concept:name"] for e in trace]
        from collections import Counter as C
        c = C(names)
        return sum(1 for k, v in c.items() if v > 1)

    stats = []
    for trace in log:
        cid = trace.attributes.get("concept:name")
        stats.append({"case_id": cid, "adjacent_repeats": count_adjacent_repeats(trace), "loops": count_return_loops(trace)})
    pd.DataFrame(stats).to_csv(os.path.join(TABLES_DIR, "rework_by_case.csv"), index=False, encoding="utf-8-sig")
    print(f"Сохранено: {os.path.join(TABLES_DIR, 'rework_by_case.csv')}")
    same_pairs = df_edges.groupby(["src", "dst"]).size().reset_index(name="count")
    repeats_same = same_pairs[same_pairs["src"] == same_pairs["dst"]].sort_values("count", ascending=False)
    repeats_same.to_csv(os.path.join(TABLES_DIR, "repeated_same_activity.csv"), index=False, encoding="utf-8-sig")
    print(f"Сохранено: {os.path.join(TABLES_DIR, 'repeated_same_activity.csv')}")

    # Тренд по месяцам
    rows = []
    for trace in log:
        cid = trace.attributes.get("concept:name")
        ts = [e["time:timestamp"] for e in trace]
        if len(ts) >= 2:
            start = min(ts)
            dur_s = (max(ts) - min(ts)).total_seconds()
            rows.append({"case_id": cid, "start_date": pd.Timestamp(start).date(), "dur_s": dur_s})
    df_time = pd.DataFrame(rows)
    if len(df_time):
        df_time["month"] = pd.to_datetime(df_time["start_date"]).dt.to_period("M").dt.to_timestamp()
        trend = df_time.groupby("month")["dur_s"].agg(["count", "median", "mean", "max"]).reset_index()
        trend.to_csv(os.path.join(TABLES_DIR, "duration_trend_by_month.csv"), index=False, encoding="utf-8-sig")
        print(f"Сохранено: {os.path.join(TABLES_DIR, 'duration_trend_by_month.csv')}")

    # SLA (если передан)
    rules = parse_sla(args.sla)
    if rules:
        df_sla = sla_breaches(df_edges, rules)
        df_sla.to_csv(os.path.join(TABLES_DIR, "sla_breaches.csv"), index=False, encoding="utf-8-sig")
        print(f"Сохранено: {os.path.join(TABLES_DIR, 'sla_breaches.csv')}")
    else:
        agg[["src", "dst", "count", "p90_s"]].to_csv(os.path.join(TABLES_DIR, "edges_sla_template.csv"), index=False, encoding="utf-8-sig")
        print(f"Сохранено: {os.path.join(TABLES_DIR, 'edges_sla_template.csv')} (добавь столбец sla_s и прогони скрипт с --sla)")

    print("\nГОТОВО. Смотри PNG в текущей директории и CSV в папке 'tables'.")


if __name__ == "__main__":
    main()
