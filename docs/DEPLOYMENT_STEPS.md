# DEPLOYMENT STEPS

**Status:**  Code uploaded to GitHub  
**Next:** Clone, configure, and deploy

---

## ⚡ **Quick Overview**

```
Time to deploy: 4-8 hours initial setup
Time to live:   1+ quarters paper trading

Steps:
1. Clone from GitHub        (5 min)
2. Setup environment         (10 min)
3. Configure                 (15 min)
4. Setup data               (1-2 hours)
5. Train models             (2-4 hours)
6. Test execution           (30 min)
7. Paper trade              (1+ quarters) ← CRITICAL
8. Go live gradually        (3-4 months scale-up)
```

---

## 📦 **STEP 1: Clone Repository (5 min)**

```bash
# Clone your repo
git clone https://github.com/LeoZhangzaolin/institutional-ownership-strategy.git
cd institutional-ownership-strategy

# Verify structure
ls
# Should see: src/ scripts/ config/ data/ logs/ reports/
```

---

## 🛠️ **STEP 2: Environment Setup (10 min)**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import pandas, numpy, sklearn; print('✓ Ready')"
```

---

## ⚙️ **STEP 3: Configuration (15 min)**

```bash
# Copy example config
cp config/config.example.yaml config/config.yaml

# Edit with your settings
nano config/config.yaml
```

**Required changes:**
```yaml
data:
  wrds_username: 'YOUR_WRDS_USERNAME'  # ← ADD YOURS

broker:
  interactive_brokers:
    port: 7497  # Paper: 7497, Live: 7496
  # OR
  alpaca:
    api_key: 'YOUR_KEY'      # ← ADD YOURS
    secret_key: 'YOUR_SECRET'  # ← ADD YOURS

trading:
  broker: 'interactive_brokers'  # or 'alpaca'
  paper_trading: true  # ← KEEP TRUE initially
  
portfolio:
  initial_capital: 1000000  # Adjust to your capital
```

---

## 💾 **STEP 4: Data Setup (1-2 hours)**

### **Option A: Use Existing Parquet (FAST - 60 min)**

```bash
# Copy your 13F data
cp -r /path/to/your/13f_parquet/* data/13f_parquet/

# Verify
ls data/13f_parquet/
# Should see: yq=2013-Q1/, yq=2013-Q2/, etc.

# Build model dataframe
python scripts/quarterly_update.py update-data
# Takes: ~60 minutes
```

### **Option B: Download from WRDS (SLOW - 4+ hours)**

```bash
# Download each quarter from WRDS
python scripts/quarterly_update.py update-13f --quarter 2025-09-30
# Repeat for all quarters (50+)

# Then build dataframe
python scripts/quarterly_update.py update-data
```

---

## 🤖 **STEP 5: Train Models (2-4 hours)**

```bash
# Train all models with walkforward validation
python scripts/quarterly_update.py train-models

# Expected:
# [Training] Starting walkforward training
# [1/15] 2022-03-31 | train=400000 test=30000 → 3 models, 8.2s
# ...
# [Training] ✓ Complete in 2.3 hours

# Verify results
python -c "
import pandas as pd
preds = pd.read_parquet('data/models/predictions.parquet')
ensemble = preds[preds['model'] == 'Ensemble']
by_q = ensemble.groupby('date_q_end').apply(lambda g: g['pred_ex'].corr(g['ret_excess_next']))
print(f'Mean IC: {by_q.mean():.3f}')
print(f'IC Sharpe: {by_q.mean() / by_q.std():.2f}')
print('Expected: IC Sharpe ~2.5-2.7')
"
```

**If Sharpe < 2.0:** Check data quality, retrain, or investigate

---

## 📊 **STEP 6: Generate Signals (5 min)**

```bash
# Generate trading signals for latest quarter
python scripts/quarterly_update.py generate-signals

# Output:
# [SIGNALS] ✓ Saved weights: data/signals/2025-09-30_weights.parquet
# [SIGNALS] Long positions: 225
# [SIGNALS] Short positions: 225
# [SIGNALS] Gross exposure: 2.00
```

---

## 🧪 **STEP 7: Test Execution (30 min)**

### **Dry Run (No Real Trades):**

```bash
# Simulate execution
python scripts/live_trading.py --dry-run

# Output:
# [REBALANCE] ⚠ PAPER TRADING MODE
# [REBALANCE] ✓ Loaded 450 target weights
# [REBALANCE] ✓ Daily limits OK
# [REBALANCE] Would submit orders:
#   1. BUY 500 AAPL @ $150
#   2. SELL 300 TSLA @ $245
# [REBALANCE] ✓ Complete
```

### **Test Monitoring:**

```bash
# Run daily monitoring
python scripts/monitor_daily.py

# Output:
# [MONITOR] ✓ Portfolio value: $1,000,000
# [MONITOR] ✓ Daily P&L: $0 (0.00%)
# [MONITOR] ✓ All risk limits passed
```

### **Test Position Verification:**

```bash
# Verify positions
python scripts/verify_positions.py

# Output:
# [VERIFY] ✓ Broker positions: 0
# [VERIFY] ✓ All positions verified
```

---

## 🚦 **STEP 8: Paper Trading (1+ QUARTERS - CRITICAL)**

⚠️ **DO NOT skip this step!**

### **Connect to Broker:**

**Interactive Brokers:**
```bash
# 1. Start TWS or IB Gateway
# 2. Set to Paper Trading mode
# 3. Enable API (port 7497)
# 4. Test connection:
python -c "
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
print('✓ Connected')
"
```

**Alpaca:**
```bash
# Test connection
python -c "
import alpaca_trade_api as tradeapi
api = tradeapi.REST('KEY', 'SECRET', 'https://paper-api.alpaca.markets')
print(f'✓ Connected: ${api.get_account().portfolio_value}')
"
```

### **Execute First Rebalance:**

```bash
# Make sure paper_trading: true in config
python scripts/live_trading.py

# Monitor
tail -f logs/trading.log
```

### **Set Up Daily Monitoring:**

```bash
# Linux/Mac: Add to cron
crontab -e
# Add: 0 17 * * 1-5 cd /path/to/project && source venv/bin/activate && python scripts/monitor_daily.py

# Windows: Use Task Scheduler
```

### **Monitor Period (MINIMUM 1 QUARTER):**

```
Week 1-4:
├── Run daily monitoring
├── Check reports daily
├── Verify positions
└── Monitor P&L

Month 2-3:
├── Track Sharpe ratio
├── Compare to backtest (should be ~2.0-2.5)
├── Verify execution quality
└── Check costs

After 1 Full Quarter:
├── Review complete cycle
├── Validate performance
├── Decision: Ready for live?
└── If yes → Proceed to Step 9
```

---

## 🎯 **STEP 9: Go Live (GRADUAL - 3-4 months)**

⚠️ **ONLY after successful paper trading**

### **Pre-Live Checklist:**

```
[ ] Paper traded successfully 1+ quarters
[ ] Sharpe close to backtest (2.0-2.7)
[ ] No execution errors
[ ] Risk limits working
[ ] Position reconciliation accurate
[ ] Comfortable with system
[ ] Capital ready
[ ] Broker live account approved
```

### **Switch to Live:**

```yaml
# Edit config.yaml
trading:
  paper_trading: false  # ← CHANGE

broker:
  interactive_brokers:
    port: 7496  # ← CHANGE from 7497
```

### **Start Small (20%):**

```yaml
portfolio:
  initial_capital: 200000  # 20% if total is $1M
```

### **Scaling Schedule:**

```
Month 1: 20% capital
├── Monitor very closely
└── If successful → 50%

Month 2: 50% capital
├── Continue monitoring
└── If successful → 75%

Month 3: 75% capital
├── Ensure consistency
└── If successful → 100%

Month 4+: 100% capital
└── Full production
```

---

## 📅 **STEP 10: Ongoing Operations**

### **Quarterly (Every 3 Months):**

```bash
# 45 days after quarter end
# Example: Q3 ends Sep 30 → Nov 15

# Complete pipeline
python scripts/quarterly_update.py full-update --quarter 2025-09-30
python scripts/live_trading.py

# OR step by step
python scripts/quarterly_update.py update-13f --quarter 2025-09-30
python scripts/quarterly_update.py update-data
python scripts/quarterly_update.py train-models
python scripts/quarterly_update.py generate-signals
python scripts/live_trading.py
```

### **Daily:**

```bash
# Automated via cron/Task Scheduler
python scripts/monitor_daily.py

# Review reports
cat reports/daily_report_$(date +%Y-%m-%d).txt
```

### **Weekly:**

```
Monday:
├── Review weekly performance
└── Check for alerts

Friday:
├── Weekly summary
└── System health check
```

---

## 🔧 **Troubleshooting**

### **WRDS Connection Failed:**
```bash
python -c "import wrds; db = wrds.Connection(); print('✓')"
# Check: username, subscription, network
```

### **Broker Connection Failed:**
```bash
# IB: TWS running? API enabled? Port 7497?
# Alpaca: API keys correct?
```

### **Training Too Slow:**
```yaml
# config.yaml
models:
  max_train_quarters: 20  # Reduce from 31
  enable_regime_models: false  # Faster
```

### **Low Sharpe:**
```bash
# Check IC by quarter
python -c "
import pandas as pd
preds = pd.read_parquet('data/models/predictions.parquet')
# Analyze...
"
```

---

## 📊 **Performance Expectations**

| Metric | Backtest | Live (Expected) |
|--------|----------|-----------------|
| Sharpe | 2.67 | 2.0-2.2 |
| Return | 25-30% | 20-25% |
| Drawdown | 12-15% | 15-20% |
| Win Rate | 55-60% | 52-57% |

**If Sharpe < 1.5:** Investigate immediately  
**If Sharpe 1.5-2.0:** Monitor for 2-3 quarters  
**If Sharpe > 2.5:** Excellent!

---

## ✅ **Quick Checklist**

```
SETUP
[ ] Clone from GitHub
[ ] Install dependencies
[ ] Configure credentials
[ ] Setup data (Option A or B)

TRAINING
[ ] Train models
[ ] Verify Sharpe ~2.5-2.7
[ ] Generate signals

TESTING
[ ] Dry-run execution
[ ] Test monitoring
[ ] Test verification

PAPER TRADING (MANDATORY)
[ ] Connect to broker
[ ] Execute first rebalance
[ ] Monitor for 1+ quarters
[ ] Verify performance

LIVE (AFTER PAPER TRADING)
[ ] Complete checklist
[ ] Start with 20%
[ ] Scale gradually
[ ] Establish workflow
```

---

## 🆘 **Key Commands**

```bash
# Full pipeline
python scripts/quarterly_update.py full-update --quarter YYYY-MM-DD

# Individual steps
python scripts/quarterly_update.py update-data
python scripts/quarterly_update.py train-models
python scripts/quarterly_update.py generate-signals

# Trading
python scripts/live_trading.py [--dry-run]

# Monitoring
python scripts/monitor_daily.py
python scripts/verify_positions.py
```

---

## 🎉 **Success Criteria**

You're ready for live when:
- ✅ Paper traded 1+ full quarters
- ✅ Sharpe ~2.0-2.5 (within 20% of backtest)
- ✅ No execution errors
- ✅ Risk limits working
- ✅ Daily monitoring automated
- ✅ Comfortable with system

**Good luck! 🚀**