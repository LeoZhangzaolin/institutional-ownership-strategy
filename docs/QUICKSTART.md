# ⚡ QUICKSTART GUIDE

**For:** Developers who want to get started quickly  
**Status:** ✅ Code on GitHub, ready to deploy

---

## 🚀 **10-Minute Setup**

```bash
# 1. Clone
git clone https://github.com/LeoZhangzaolin/institutional-ownership-strategy.git
cd institutional-ownership-strategy

# 2. Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp config/config.example.yaml config/config.yaml
# Edit config.yaml: Add WRDS username, broker credentials

# 4. Setup data (choose one)
# A. Use existing parquet (FAST)
cp -r /your/13f_parquet/* data/13f_parquet/
# B. Download from WRDS (SLOW)
python scripts/quarterly_update.py update-13f --quarter 2025-09-30

# 5. Build & train
python scripts/quarterly_update.py update-data        # 60 min
python scripts/quarterly_update.py train-models       # 2-4 hours

# 6. Test
python scripts/live_trading.py --dry-run

# ✓ Ready for paper trading!
```

---

## 📋 **Key Commands**

### **Data Pipeline:**
```bash
# Download 13F
python scripts/quarterly_update.py update-13f --quarter YYYY-MM-DD

# Build dataframe
python scripts/quarterly_update.py update-data

# Train models
python scripts/quarterly_update.py train-models

# Generate signals
python scripts/quarterly_update.py generate-signals

# Complete pipeline
python scripts/quarterly_update.py full-update --quarter YYYY-MM-DD
```

### **Trading:**
```bash
# Dry run (test)
python scripts/live_trading.py --dry-run

# Paper trading
python scripts/live_trading.py  # with paper_trading: true

# Live trading
python scripts/live_trading.py  # with paper_trading: false
```

### **Monitoring:**
```bash
# Daily monitoring
python scripts/monitor_daily.py

# Position verification
python scripts/verify_positions.py

# Check logs
tail -f logs/trading.log
cat reports/daily_report_$(date +%Y-%m-%d).txt
```

---

## ⚙️ **Configuration**

### **Minimal config.yaml:**
```yaml
data:
  wrds_username: 'YOUR_USERNAME'

broker:
  interactive_brokers:
    host: '127.0.0.1'
    port: 7497  # Paper: 7497, Live: 7496
    client_id: 1

trading:
  broker: 'interactive_brokers'
  paper_trading: true
  order_type: 'market'
  time_in_force: 'DAY'

portfolio:
  initial_capital: 1000000

risk:
  max_daily_loss_pct: 0.02
  max_drawdown_pct: 0.15
```

---

## 📊 **Data Options**

### **Option A: Use Existing Parquet (Recommended)**
```bash
# Copy your 13F data
cp -r /your/13f_parquet/* data/13f_parquet/

# Verify
ls data/13f_parquet/
# Should see: yq=2013-Q1/, yq=2013-Q2/, ...

# Build dataframe
python scripts/quarterly_update.py update-data
```

### **Option B: Download from WRDS**
```bash
# Download quarters (slow)
python scripts/quarterly_update.py update-13f --quarter 2025-09-30

# Build dataframe
python scripts/quarterly_update.py update-data
```

---

## 🧪 **Testing**

```bash
# 1. Test configuration
python -c "from src.utils import load_config; load_config('config/config.yaml'); print('✓')"

# 2. Test data loading
python -c "import pandas as pd; df = pd.read_parquet('data/cache/model_df.parquet'); print(f'✓ {len(df):,} rows')"

# 3. Test broker connection
python -c "from src.order_execution import OrderExecutor; from src.utils import load_config; executor = OrderExecutor(load_config('config/config.yaml')); print('✓')"

# 4. Test execution (dry run)
python scripts/live_trading.py --dry-run

# 5. Test monitoring
python scripts/monitor_daily.py
```

---

## 📅 **Workflows**

### **Initial Setup (One-time):**
```
1. Clone from GitHub         (5 min)
2. Install dependencies       (10 min)
3. Configure                  (15 min)
4. Setup data                 (1-2 hours)
5. Train models              (2-4 hours)
6. Test execution            (30 min)
   ↓
Ready for paper trading
```

### **Quarterly Rebalance:**
```
Every 3 months (45 days after quarter end):

1. python scripts/quarterly_update.py update-13f --quarter YYYY-MM-DD
2. python scripts/quarterly_update.py update-data
3. python scripts/quarterly_update.py train-models
4. python scripts/quarterly_update.py generate-signals
5. python scripts/live_trading.py

OR: python scripts/quarterly_update.py full-update --quarter YYYY-MM-DD
    python scripts/live_trading.py
```

### **Daily Operations:**
```
Automated (cron/Task Scheduler):
- python scripts/monitor_daily.py (5 PM daily)

Manual:
- Check reports/daily_report_YYYY-MM-DD.txt
- Review logs/trading.log
- Verify positions if needed
```

---

## 🎯 **Strategy Overview**

### **What it does:**
- Uses SEC 13F filings (institutional ownership data)
- Identifies skilled managers using 8Q trailing performance
- Creates skill-weighted ownership features
- Trains 12 models (4 regimes × 3 types)
- Inverse-variance weighted ensemble
- Quarterly rebalancing (dollar-neutral, long-short)

### **Performance:**
- Backtest Sharpe: 2.67 (2022-2025)
- Expected Live: 2.0-2.2
- Rebalance: Quarterly (after 13F data available)
- Universe: ~2,500 stocks
- Positions: ~450 (225 long, 225 short)

### **Data Sources:**
- 13F filings: tr_13f.s34 (quarterly)
- CRSP: msf, msenames (daily/monthly)
- S&P 500: wrds.comp.idx_index
- Risk-free rate: wrds.frb.rates_daily

---

## 🔧 **Troubleshooting**

### **Import Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### **WRDS Connection Failed:**
```bash
# Test connection
python -c "import wrds; db = wrds.Connection(); print('✓')"

# Check:
# - Username correct in config
# - WRDS subscription active
# - Network/firewall
```

### **Broker Connection Failed:**
```bash
# IB: Check TWS running, API enabled, port 7497
# Alpaca: Check API keys in config

# Test:
from src.order_execution import OrderExecutor
from src.utils import load_config
config = load_config('config/config.yaml')
executor = OrderExecutor(config)
```

### **Training Takes Forever:**
```yaml
# Reduce training time in config.yaml:
models:
  max_train_quarters: 20  # Reduce from 31
  enable_regime_models: false  # Faster
```

### **Low Sharpe Ratio:**
```bash
# Check predictions
python -c "
import pandas as pd
preds = pd.read_parquet('data/models/predictions.parquet')
ensemble = preds[preds['model'] == 'Ensemble']
ic = ensemble.groupby('date_q_end').apply(lambda g: g['pred_ex'].corr(g['ret_excess_next']))
print(f'Mean IC: {ic.mean():.3f}')
print(f'IC Sharpe: {ic.mean()/ic.std():.2f}')
"

# If < 2.0:
# - Check data quality
# - Retrain models
# - Verify features
```

---

## 📁 **Project Structure**

```
institutional-ownership-strategy/
├── config/
│   ├── config.yaml           # Your configuration (gitignored)
│   └── config.example.yaml   # Template
├── src/
│   ├── data_pipeline.py      # 13F + CRSP data
│   ├── feature_engineering.py # Feature creation
│   ├── regime_detection.py   # Market regime
│   ├── model_training.py     # ML models
│   ├── portfolio_optimization.py # Portfolio construction
│   ├── order_execution.py    # Trade execution
│   ├── risk_management.py    # Risk limits
│   └── utils.py              # Utilities
├── scripts/
│   ├── quarterly_update.py   # Data + training pipeline
│   ├── live_trading.py       # Execute rebalancing
│   ├── monitor_daily.py      # Daily monitoring
│   └── verify_positions.py   # Position reconciliation
├── data/                     # Data files (gitignored)
│   ├── 13f_parquet/          # Raw 13F data
│   ├── cache/                # model_df.parquet
│   ├── models/               # Trained models
│   └── signals/              # Trading signals
├── logs/                     # Log files (gitignored)
└── reports/                  # Daily reports (gitignored)
```

---

## 🚦 **Deployment Path**

```
Phase 1: Setup (4-8 hours)
├── Clone + install
├── Configure
├── Setup data
└── Train models

Phase 2: Testing (30 min)
├── Dry run
├── Test monitoring
└── Verify everything works

Phase 3: Paper Trading (1+ quarters) ← CRITICAL
├── Execute in paper mode
├── Monitor daily
└── Validate performance

Phase 4: Live (Gradual)
├── Month 1: 20% capital
├── Month 2: 50% capital
├── Month 3: 75% capital
└── Month 4+: 100% capital
```

---

## ✅ **Pre-Live Checklist**

```
[ ] Paper traded 1+ quarters successfully
[ ] Sharpe ~2.0-2.5 (close to backtest)
[ ] No execution errors
[ ] Risk limits working correctly
[ ] Daily monitoring automated
[ ] Position reconciliation accurate
[ ] Comfortable with system behavior
[ ] Capital allocated
[ ] Broker live account approved
```

---

## 📊 **Key Metrics**

| Metric | Target |
|--------|--------|
| IC (Information Coefficient) | 0.05-0.08 |
| IC Sharpe | 2.0-2.7 |
| Portfolio Sharpe | 2.0-2.2 (live) |
| Positions | ~450 (225L/225S) |
| Gross Exposure | 2.0 |
| Net Exposure | ~0% (dollar-neutral) |
| Turnover | Quarterly |
| Fill Rate | >85% |

---

## 🆘 **Quick Help**

```bash
# Check status
python -c "
from pathlib import Path
print('Data:', 'YES' if (Path('data/cache/model_df.parquet')).exists() else 'NO')
print('Models:', 'YES' if (Path('data/models/models.pkl')).exists() else 'NO')
print('Signals:', 'YES' if list(Path('data/signals').glob('*.parquet')) else 'NO')
"

# View logs
tail -f logs/trading.log

# Latest report
cat reports/daily_report_$(date +%Y-%m-%d).txt

# Check performance
python -c "
import pandas as pd
hist = pd.read_csv('logs/portfolio_history.csv')
print(hist.tail())
"
```

---

## 🔗 **Resources**

- **DEPLOYMENT_STEPS.md** - Detailed deployment guide
- **README.md** - Project overview
- **docs/** - Additional documentation

---

## 🎉 **Quick Start Summary**

```bash
# 1. Clone & setup (15 min)
git clone <YOUR_REPO>
cd institutional-ownership-strategy
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure (5 min)
cp config/config.example.yaml config/config.yaml
nano config/config.yaml  # Add credentials

# 3. Data & training (3-6 hours)
cp -r /your/13f_parquet/* data/13f_parquet/
python scripts/quarterly_update.py update-data
python scripts/quarterly_update.py train-models

# 4. Test (5 min)
python scripts/live_trading.py --dry-run

# 5. Paper trade (1+ quarters)
python scripts/live_trading.py

# ✓ Ready!
```

**Good luck! 🚀**