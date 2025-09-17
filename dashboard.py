import os
import subprocess
import time
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(page_title="Process Mining Dashboard", layout="wide")

PRIMARY_BG = "#0b1020"
CARD_BG = "#11162a"
ACCENT = "#4f46e5"

st.markdown(
    f"""
    <style>
    .main {{background: {PRIMARY_BG}; color: #e8ecf6;}}
    .stMarkdown, .stDataFrame, .stMetric, .stTable {{color: #e8ecf6 !important;}}
    .css-1cpxqw2 a {{color: {ACCENT} !important;}}
    .block-container {{padding-top: 1rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    try:
        if os.path.isfile(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def load_text(path: str) -> str:
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

st.sidebar.header("Настройки")
base_dir = st.sidebar.text_input("База проекта", value=os.getcwd())
tables_dir = st.sidebar.text_input("Папка с CSV", value=os.path.join(base_dir, "tables"))
html_path = st.sidebar.text_input("Интерактивный граф (HTML)", value=os.path.join(base_dir, "dfg_combined.html"))

paths = {
    'sla_breaches_cases': os.path.join(tables_dir, 'sla_breaches_cases.csv'),
    'stage_kpi': os.path.join(tables_dir, 'stage_kpi.csv'),
    'handoff_matrix': os.path.join(tables_dir, 'handoff_matrix.csv'),
    'ping_pong_pairs': os.path.join(tables_dir, 'ping_pong_pairs.csv'),
    'variant_explorer': os.path.join(tables_dir, 'variant_explorer.csv'),
    'anomalies_cases': os.path.join(tables_dir, 'anomalies_cases.csv'),
    'cases_durations': os.path.join(tables_dir, 'cases_durations.csv'),
    'returns': os.path.join(tables_dir, 'returns.csv'),
    'return_to_start': os.path.join(tables_dir, 'return_to_start.csv'),
    'activity_ping_pong': os.path.join(tables_dir, 'activity_ping_pong.csv'),
    'rare_activities': os.path.join(tables_dir, 'rare_activities.csv'),
    'manual_steps': os.path.join(tables_dir, 'manual_steps.csv'),
    'unsuccess_outcomes': os.path.join(tables_dir, 'unsuccess_outcomes.csv'),
    'cohort_trends': os.path.join(tables_dir, 'cohort_trends.csv'),
    'duration_trend_test': os.path.join(tables_dir, 'duration_trend_test.txt'),
}

D: dict[str, pd.DataFrame | str | None] = {}
for k, p in paths.items():
    D[k] = load_csv(p) if p.endswith('.csv') else load_text(p)

st.title("Process Mining — Dashboard")
st.caption("Просмотр метрик, отклонений и нарушений SLA. Укажите путь к интерактивному графу, чтобы видеть DFG.")

# KPI шапка
col1, col2, col3, col4 = st.columns(4)
cd = D.get('cases_durations')
if isinstance(cd, pd.DataFrame) and not cd.empty:
    arr = cd['case_duration_s'].dropna().to_numpy()
    col1.metric("Кейсов", f"{len(arr):,}".replace(',', ' '))
    col2.metric("Median (с)", f"{np.median(arr):.0f}")
    col3.metric("p90 (с)", f"{np.percentile(arr, 90):.0f}")
    col4.metric("Max (с)", f"{np.max(arr):.0f}")
else:
    col1.metric("Кейсов", "—")

# Вкладки
tabs = st.tabs([
    "DFG",
    "Stages",
    "Handoff",
    "SLA",
    "Variants",
    "Anomalies",
    "Returns",
    "Rare/Manual",
    "Cohorts",
    "Trend test",
])

with tabs[0]:
    st.subheader("Интерактивный граф процесса (DFG)")

    with st.expander("Собрать/обновить граф", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            input_csv = st.text_input("Входной CSV", value=os.path.join(base_dir, "case_championship_last.csv"))
            case_col = st.text_input("Колонка кейса", value="ID")
            activity_col = st.text_input("Колонка активности", value="Событие")
            ts_col = st.text_input("Колонка времени", value="Время")
            ts_format = st.text_input("Формат времени (опционально)", value="")
            rankdir = st.selectbox("Направление", options=["TB","LR","BT","RL"], index=0)
            sep = st.text_input("Разделитель CSV", value=",")
            encoding = st.text_input("Кодировка", value="utf-8")
        with colB:
            min_freq = st.number_input("Мин. частота ребра", min_value=0, max_value=1000, value=1, step=1)
            rare_edge_threshold = st.number_input("Порог редких рёбер", min_value=0, max_value=1000, value=3, step=1)
            bottleneck_p90_threshold_s = st.number_input("Порог p90 (сек) для bottleneck", min_value=0, max_value=864000, value=43200, step=3600)
            sla_csv = st.text_input("SLA CSV", value=os.path.join(tables_dir, "edges_sla_template.csv"))
            top_variant_csv = st.text_input("Top variants CSV", value=os.path.join(tables_dir, "variants_top.csv"))
            rare_activity_threshold = st.number_input("Порог редких активностей (доля)", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
        out_png = os.path.join(base_dir, "dfg_combined.png")

        if st.button("Собрать граф", type="primary"):
            cmd = [
                "/opt/anaconda3/bin/python",
                os.path.join(base_dir, "process_report.py"),
                "-i", input_csv,
                "-o", out_png,
                "--case-col", case_col,
                "--activity-col", activity_col,
                "--timestamp-col", ts_col,
                "--rankdir", rankdir,
                "--min-freq", str(min_freq),
                "--sep", sep,
                "--encoding", encoding,
                "--sla-csv", sla_csv,
                "--top-variant-csv", top_variant_csv,
                "--rare-edge-threshold", str(rare_edge_threshold),
                "--bottleneck-p90-threshold-s", str(int(bottleneck_p90_threshold_s)),
                "--tables-dir", tables_dir,
                "--rare-activity-threshold", str(rare_activity_threshold),
            ]
            if ts_format.strip():
                cmd.extend(["--timestamp-format", ts_format.strip()])
            try:
                with st.spinner("Генерация графа..."):
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    st.success("Граф сгенерирован")
                    st.code(proc.stdout or "", language="bash")
                    # После генерации обновим HTML
                    html_target = os.path.join(base_dir, "dfg_combined.html")
                    if os.path.isfile(html_target):
                        load_text.clear()  # очистить кэш
                        # небольшая пауза для файловой системы
                        time.sleep(0.2)
                        html_content = load_text(html_target)
                        st.components.v1.html(html_content, height=800, scrolling=True)
                        html_path = html_target
                    else:
                        st.warning("HTML граф не найден. Проверьте логи генерации.")
            except subprocess.CalledProcessError as e:
                st.error("Ошибка генерации графа")
                st.code((e.stdout or "") + "\n" + (e.stderr or ""))

    if os.path.isfile(html_path):
        html = load_text(html_path)
        st.components.v1.html(html, height=800, scrolling=True)
        st.info("Для обновления параметров используйте блок 'Собрать/обновить граф'.")
    else:
        st.warning("Файл dfg_combined.html не найден. Запустите сборку графа выше или укажите путь в сайдбаре.")

with tabs[1]:
    st.subheader("KPI по стадиям")
    df = D.get('stage_kpi')
    if isinstance(df, pd.DataFrame) and not df.empty:
        topN = st.slider("Top N стадий", 5, 100, 20, key="stages_topn")
        top_by_p90 = df.sort_values('p90_s', ascending=False).head(topN)
        st.dataframe(top_by_p90, use_container_width=True)
        st.bar_chart(top_by_p90.set_index('stage')['p90_s'])
        st.bar_chart(top_by_p90.set_index('stage')['time_share'])
        # Доп. графики
        c1 = alt.Chart(top_by_p90).mark_bar(color='#f59e0b').encode(x='p90_s:Q', y=alt.Y('stage:N', sort='-x')).properties(height=400)
        c2 = alt.Chart(df.sort_values('p50_s', ascending=False).head(topN)).mark_bar(color='#38bdf8').encode(x='p50_s:Q', y=alt.Y('stage:N', sort='-x')).properties(height=400)
        c3 = alt.Chart(df.sort_values('count', ascending=False).head(topN)).mark_bar(color='#22c55e').encode(x='count:Q', y=alt.Y('stage:N', sort='-x')).properties(height=400)
        st.altair_chart(c1, use_container_width=True)
        st.altair_chart(c2, use_container_width=True)
        st.altair_chart(c3, use_container_width=True)
    else:
        st.info("Нет данных stage_kpi.csv")

with tabs[2]:
    st.subheader("Передачи и пинг‑понг")
    hand = D.get('handoff_matrix')
    pp = D.get('ping_pong_pairs')
    topN = st.slider("Top N пар", 5, 100, 20, key="handoff_topn")
    if isinstance(hand, pd.DataFrame) and not hand.empty:
        st.dataframe(hand.sort_values('count', ascending=False).head(50), use_container_width=True)
        st.altair_chart(
            alt.Chart(hand.sort_values('count', ascending=False).head(topN)).mark_bar(color='#60a5fa')
            .encode(x='count:Q', y=alt.Y('to:N', sort='-x'), color=alt.Color('from:N', legend=None)).properties(height=450),
            use_container_width=True
        )
        # Heatmap
        hm = hand.copy().groupby(['from','to'])['count'].sum().reset_index()
        st.altair_chart(
            alt.Chart(hm).mark_rect().encode(
                x=alt.X('from:N', sort='-y'), y=alt.Y('to:N', sort='-x'), color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'))
            ).properties(height=500), use_container_width=True
        )
    if isinstance(pp, pd.DataFrame) and not pp.empty:
        st.dataframe(pp.sort_values('ping_pongs', ascending=False).head(50), use_container_width=True)
        st.altair_chart(
            alt.Chart(pp.sort_values('ping_pongs', ascending=False).head(topN)).mark_bar(color='#a78bfa')
            .encode(x='ping_pongs:Q', y=alt.Y('a:N', sort='-x'), color=alt.Color('b:N', legend=None)).properties(height=400),
            use_container_width=True
        )

with tabs[3]:
    st.subheader("SLA-нарушения")
    sla = D.get('sla_breaches_cases')
    if isinstance(sla, pd.DataFrame) and not sla.empty:
        grouped = sla.groupby(['src','dst']).size().reset_index(name='breaches').sort_values('breaches', ascending=False)
        st.dataframe(grouped.head(100), use_container_width=True)
        topN = st.slider("Top N переходов", 5, 100, 20, key="sla_topn")
        st.altair_chart(
            alt.Chart(grouped.head(topN)).mark_bar(color='#ef4444').encode(x='breaches:Q', y=alt.Y('dst:N', sort='-x'), color=alt.Color('src:N', legend=None)).properties(height=500),
            use_container_width=True
        )
        # Распределение превышений
        st.altair_chart(
            alt.Chart(sla).mark_boxplot(color='#ef4444').encode(x=alt.X('src:N', title='Источник'), y=alt.Y('delta_s:Q', title='Delta (сек)'), color='dst:N').properties(height=500),
            use_container_width=True
        )
    else:
        st.info("Нет данных sla_breaches_cases.csv")

with tabs[4]:
    st.subheader("Варианты (Variant Explorer)")
    ve = D.get('variant_explorer')
    cd = D.get('cases_durations')
    thr = None
    if isinstance(cd, pd.DataFrame) and not cd.empty:
        thr = float(np.percentile(cd['case_duration_s'].dropna().to_numpy(), 90))
        st.caption(f"Порог медленных по p90 кейса: {thr:.0f} с")
    topN = st.slider("Top N вариантов", 5, 100, 20, key="variants_topn")
    if isinstance(ve, pd.DataFrame) and not ve.empty:
        st.dataframe(ve.sort_values(['count','median_s'], ascending=[False, True]).head(topN), use_container_width=True)
        st.altair_chart(
            alt.Chart(ve.sort_values('count', ascending=False).head(topN)).mark_bar(color='#34d399').encode(x='count:Q', y=alt.Y('variant:N', sort='-x')).properties(height=500),
            use_container_width=True
        )
        if thr is not None:
            slow = ve[ve['median_s'] >= thr].sort_values(['count','median_s'], ascending=[False, False]).head(topN)
            st.altair_chart(
                alt.Chart(slow).mark_bar(color='#fbbf24').encode(x='median_s:Q', y=alt.Y('variant:N', sort='-x'), color=alt.Color('top_channel:N', legend=None)).properties(height=500),
                use_container_width=True
            )

with tabs[5]:
    st.subheader("Аномалии (>p99)")
    an = D.get('anomalies_cases')
    if isinstance(an, pd.DataFrame) and not an.empty:
        topN = st.slider("Top N аномалий", 5, 100, 20, key="anom_topn")
        st.dataframe(an.sort_values('duration_s', ascending=False).head(100), use_container_width=True)
        # Гистограмма длительностей
        st.altair_chart(alt.Chart(an).mark_bar(color='#60a5fa').encode(x=alt.X('duration_s:Q', bin=alt.Bin(maxbins=30)), y='count()', tooltip=['count()']).properties(height=300), use_container_width=True)
        # Лупы
        st.altair_chart(alt.Chart(an).mark_bar(color='#a78bfa').encode(x=alt.X('loops:Q', bin=alt.Bin(maxbins=15)), y='count()').properties(height=300), use_container_width=True)
        # Пинг-понг
        st.altair_chart(alt.Chart(an).mark_bar(color='#f472b6').encode(x=alt.X('ping_pong:Q', bin=alt.Bin(maxbins=15)), y='count()').properties(height=300), use_container_width=True)
        # Связь длительности и лупов
        st.altair_chart(alt.Chart(an.head(topN)).mark_circle(size=80, color='#22c55e').encode(x='loops:Q', y='duration_s:Q', tooltip=['case_id','duration_s','loops','ping_pong']).properties(height=400), use_container_width=True)
    else:
        st.info("Нет данных anomalies_cases.csv")

with tabs[6]:
    st.subheader("Возвраты и A→B→A")
    ret = D.get('returns')
    rts = D.get('return_to_start')
    app = D.get('activity_ping_pong')
    topN = st.slider("Top N переходов", 5, 100, 20, key="returns_topn")
    if isinstance(ret, pd.DataFrame) and not ret.empty:
        st.dataframe(ret.head(50), use_container_width=True)
        st.altair_chart(alt.Chart(ret.sort_values('count', ascending=False).head(topN)).mark_bar(color='#38bdf8').encode(x='count:Q', y=alt.Y('dst:N', sort='-x'), color=alt.Color('src:N', legend=None)).properties(height=400), use_container_width=True)
    if isinstance(rts, pd.DataFrame) and not rts.empty:
        st.dataframe(rts.head(50), use_container_width=True)
        st.altair_chart(alt.Chart(rts.sort_values('count', ascending=False).head(topN)).mark_bar(color='#fca5a5').encode(x='count:Q', y=alt.Y('dst:N', sort='-x'), color=alt.Color('src:N', legend=None)).properties(height=300), use_container_width=True)
    if isinstance(app, pd.DataFrame) and not app.empty:
        st.dataframe(app.head(50), use_container_width=True)
        st.altair_chart(alt.Chart(app.sort_values('count', ascending=False).head(topN)).mark_bar(color='#f472b6').encode(x='count:Q', y=alt.Y('a:N', sort='-x'), color=alt.Color('b:N', legend=None)).properties(height=300), use_container_width=True)

with tabs[7]:
    st.subheader("Редкие и ручные шаги")
    rare = D.get('rare_activities')
    man = D.get('manual_steps')
    topN = st.slider("Top N активностей", 5, 100, 20, key="rare_topn")
    if isinstance(rare, pd.DataFrame) and not rare.empty:
        st.dataframe(rare.sort_values('share').head(100), use_container_width=True)
        st.altair_chart(alt.Chart(rare.sort_values('share').head(topN)).mark_bar(color='#fde68a').encode(x='share:Q', y=alt.Y('activity:N', sort='x')).properties(height=450), use_container_width=True)
    if isinstance(man, pd.DataFrame) and not man.empty:
        cols = st.columns(2)
        cols[0].altair_chart(alt.Chart(man.sort_values('manual_channel_events', ascending=False).head(topN)).mark_bar(color='#fb923c').encode(x='manual_channel_events:Q', y=alt.Y('activity:N', sort='-x')).properties(height=400), use_container_width=True)
        cols[1].altair_chart(alt.Chart(man.sort_values('worker_missing_events', ascending=False).head(topN)).mark_bar(color='#94a3b8').encode(x='worker_missing_events:Q', y=alt.Y('activity:N', sort='-x')).properties(height=400), use_container_width=True)

with tabs[8]:
    st.subheader("Коорт‑тренды")
    coh = D.get('cohort_trends')
    if isinstance(coh, pd.DataFrame) and not coh.empty:
        seg = st.selectbox("Сегмент", options=sorted(coh['segment'].dropna().unique().tolist())) if 'segment' in coh.columns else None
        dfc = coh.copy()
        if seg:
            dfc = dfc[dfc['segment'] == seg]
        try:
            dfc['month'] = pd.to_datetime(dfc['month'])
        except Exception:
            pass
        st.line_chart(data=dfc.sort_values('month').set_index('month')[['median_s','p90_s']])
        # Доп. графики: bar по месяцам
        st.altair_chart(alt.Chart(dfc).mark_bar(color='#60a5fa').encode(x='month:T', y='median_s:Q').properties(height=300), use_container_width=True)
        st.altair_chart(alt.Chart(dfc).mark_bar(color='#f59e0b').encode(x='month:T', y='p90_s:Q').properties(height=300), use_container_width=True)
        st.dataframe(dfc.sort_values(['segment','month']), use_container_width=True)

with tabs[9]:
    st.subheader("Тест тренда (Kendall)")
    txt = D.get('duration_trend_test') or ''
    if txt:
        st.code(txt, language='text')
    else:
        st.info("Нет данных duration_trend_test.txt") 