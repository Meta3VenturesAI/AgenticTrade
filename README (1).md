# 📊 Investment Signals SaaS Platform

An integrated, automated platform to generate, score, backtest, report, and distribute investment signals using live options data, machine learning, and financial logic. Designed for high-frequency insights, professional-grade backtesting, AI scoring, and daily tactical alerts.

---

## 🔧 Core Modules and Capabilities

| Module | Description |
|--------|-------------|
| 🧠 `signals_engine.py` | Generates signals using options flow, IV, volume, sentiment |
| 📊 `auto_daily_report.py` | Automates daily signal generation + report creation |
| 📈 `backtest_signals.py` | Evaluates signal performance with full metrics |
| 📤 `telegram_alert_bot_with_pdf.py` | Sends latest signals + reports to Telegram |
| 🧠 `optimize_signals.py` | Runs grid search over thresholds (IV, volume, sentiment) |
| 📦 `database.py` + `models.py` + `schemas.py` | Full backend schema, ORM, and Pydantic models |
| 🌐 `main.py` (FastAPI) | REST API for users, signals, backtests, alerts, reports |
| 📺 `investment_dashboard_with_signals.py` | Streamlit-based UI for viewing insights and graphs |
| 🔁 `daily_signal_alert.sh` | Daily automation script for scheduling full pipeline |

---

## 🗃️ PostgreSQL Database Schema (via `schema.sql`)

Tables:
- `users` – Registered users, roles
- `signals` – All generated trading signals
- `backtests` – Historical returns per signal
- `alerts` – Sent Telegram/Email notifications
- `reports` – Links to generated PDFs/images
- `subscriptions` – User plan/billing
- `settings`, `optimizations`, `activity_log` – Custom config & performance

All mirrored in SQLAlchemy via `models.py`.

---

## 🚀 FastAPI Backend (`main.py`)

### API Coverage:

| Route | Description |
|-------|-------------|
| `/users/` | Create/list users |
| `/users/{id}` | Fetch user by ID |
| `/signals/` | Create/list signals |
| `/signals/by_user/{user_id}` | User-specific signals |
| `/backtests/`, `/backtests/by_signal/{id}` | Store/view backtests |
| `/reports/` | Upload/view reports |
| `/alerts/` | Log alert dispatch |
| `/subscriptions/` | Create/list subscriptions |

Launch:
```bash
uvicorn main:app --reload
```
Docs at: `http://localhost:8000/docs`

---

## 🧪 Signal Optimization

Run:
```bash
python optimize_signals.py
```

Outputs:
- `optimization_results.csv`
- `optimization_heatmap_avg_return.png`

---

## 🔁 Full Daily Automation

```bash
bash daily_signal_alert.sh
```

Pipeline:
- `auto_daily_report.py` → signals_log
- `backtest_signals.py` → returns
- `telegram_alert_bot_with_pdf.py` → send report

Use `crontab` to schedule daily runs:
```bash
0 8 * * * /path/to/daily_signal_alert.sh
```

---

## 📥 Example Output Files

| File | Description |
|------|-------------|
| `signals_log.csv` | Historical signals |
| `Tactical_Signals_Report.pdf` | Daily PDF |
| `backtest_report.pdf` | Strategy returns |
| `optimization_results.csv` | Grid search results |
| `return_distribution.png` | Histogram |
| `telegram_alert_bot.py` | Sends live alerts via Telegram |

---

## 📊 Streamlit Dashboard (UI)

```bash
streamlit run investment_dashboard_with_signals.py
```

Includes:
- Graphs: IV vs Price, Sentiment trend
- Tabs for signals, optimization, alerts
- Refresh + PDF export

---

## 📦 Deployment Options

- Local: `uvicorn`, `cron`, `streamlit`
- Cloud: Render, Supabase, Railway, EC2
- Docker: build container with backend + UI

---

Built for analysts, strategists, and signal-driven funds.  
From signal to strategy to automation – all in one platform.---

## 🔁 Live Data Integration

The platform includes a modular `live_data_fetcher.py` script that supports live updates from multiple sources:

### Supported Sources:
- `options` → Fetch option chain (IV, OI, Volume)
- `sentiment` → Pull Reddit, Twitter, and Analyst sentiment
- `macro` → Load macroeconomic indicators from FRED or similar

### Usage:

```bash
python live_data_fetcher.py options
python live_data_fetcher.py sentiment
python live_data_fetcher.py macro
```

These are integrated into the Streamlit dashboard with per-tab refresh buttons and an optional time-based auto-sync setting.

---

## ⏱️ Scheduling (CRON/Streamlit Cloud)

To automate data refresh:

### Example CRON setup (every 30 minutes):

```bash
*/30 * * * * /usr/bin/python3 /path/to/live_data_fetcher.py options
```

### Example Bash Script (daily run):

```bash
#!/bin/bash
python live_data_fetcher.py options
python live_data_fetcher.py sentiment
python live_data_fetcher.py macro
```

You can also use GitHub Actions, Streamlit Cloud scheduler or Airflow for automation.