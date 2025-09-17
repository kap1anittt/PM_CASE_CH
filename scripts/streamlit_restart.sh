#!/bin/bash
set -euo pipefail

PORT="${1:-8501}"
PYBIN="${PYBIN:-/opt/anaconda3/bin}"
APP_PATH="${APP_PATH:-/Users/anna/Documents/r2/PM_CASE_CH/dashboard.py}"

echo "[streamlit_restart] Target port: ${PORT}"

# Мягкое завершение процессов, слушающих порт
PIDS=$(lsof -ti tcp:${PORT} || true)
if [ -n "${PIDS}" ]; then
  echo "[streamlit_restart] Killing (TERM) PIDs on port ${PORT}: ${PIDS}"
  kill ${PIDS} 2>/dev/null || true
  sleep 1
fi

# Принудительное завершение, если ещё висят
PIDS2=$(lsof -ti tcp:${PORT} || true)
if [ -n "${PIDS2}" ]; then
  echo "[streamlit_restart] Killing (KILL) PIDs on port ${PORT}: ${PIDS2}"
  kill -9 ${PIDS2} 2>/dev/null || true
  sleep 1
fi

# Дополнительно завершим возможные процессы streamlit dashboard по сигнатуре
pkill -f "streamlit.*${APP_PATH}" 2>/dev/null || true
sleep 1

# Запуск Streamlit
echo "[streamlit_restart] Starting Streamlit on port ${PORT}"
"${PYBIN}/streamlit" run "${APP_PATH}" --server.headless true --server.port "${PORT}" 