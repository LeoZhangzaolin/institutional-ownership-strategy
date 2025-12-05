# 🚀 INSTITUTIONAL OWNERSHIP TRADING STRATEGY
## Production-Ready System - Sharpe 2.67

**Complete production system for deploying institutional ownership strategy to live trading.**

**Status:** ✅ All code on GitHub, ready to clone and deploy

---

## 🔗 QUICK NAVIGATION

| What You Need | Where to Go |
|---------------|-------------|
| 🚀 **Get started fast** | [QUICKSTART.md](docs/QUICKSTART.md) - Commands & 10-min setup |
| 📖 **Complete deployment** | [DEPLOYMENT_STEPS.md](docs/DEPLOYMENT_STEPS.md) - Step-by-step guide |
| 📊 **Strategy details** | This README - Architecture & overview |
| 🔧 **Configuration** | [config/config.example.yaml](config/config.example.yaml) - Settings template |

---

## 📦 WHAT IN IT

**Core Modules (9 files, 3,015 lines):**
- ✅ `src/data_pipeline.py` - 13F + CRSP data processing (525 lines)
- ✅ `src/feature_engineering.py` - Feature creation (300 lines)
- ✅ `src/regime_detection.py` - Market regime detection (150 lines)
- ✅ `src/variance_tracker.py` - Ensemble weighting (150 lines)
- ✅ `src/portfolio_optimization.py` - Portfolio construction (150 lines)
- ✅ `src/model_training.py` - 12-model walkforward training (720 lines)
- ✅ `src/order_execution.py` - Trade execution (520 lines)
- ✅ `src/risk_management.py` - Risk limits & tracking (450 lines)
- ✅ `src/utils.py` - Utilities & config (100 lines)

**Execution Scripts (4 files, 1,400 lines):**
- ✅ `scripts/quarterly_update.py` - Complete pipeline (520 lines)
- ✅ `scripts/live_trading.py` - Execution controller (220 lines)
- ✅ `scripts/monitor_daily.py` - Daily monitoring (380 lines)
- ✅ `scripts/verify_positions.py` - Position reconciliation (280 lines)

**Data (historical 13F data):**
- ✅ 13F_parquet - Ready to download

**Configuration & Docs:**
- ✅ `config/config.example.yaml` - Configuration template
- ✅ `requirements.txt` - All dependencies
- ✅ `DEPLOYMENT_STEPS.md` - Complete deployment guide
- ✅ `QUICKSTART.md` - Quick reference
- ✅ `README.md` - This file

**Total:** 17 files, 4,415+ lines of production code

---

## 🎯 My STRATEGY AT A GLANCE

**Performance:** Sharpe 2.67 (out-of-sample 2022-2025)

**Architecture:**
- **12 Models:** 4 regimes × 3 models (ElasticNet, RandomForest, XGBoost)
- **Ensemble:** Inverse-variance weighted
- **Features:** Skill-weighted institutional ownership + interactions + nonlinear + ranks
- **Optimization:** Regime-adaptive with turnover penalties
- **Rebalancing:** QUARTERLY (not daily/monthly!)

**Data Sources:**
- 13F Holdings: tr_13f.s34 (partitioned parquet)
- CRSP: msf, msenames (queried fresh)
- S&P 500: For regime detection

---

## ⚡ QUICK START

### 1. Clone from GitHub (5 min)

```bash
# Clone your repository
git clone https://github.com/LeoZhangzaolin/institutional-ownership-strategy.git
cd institutional-ownership-strategy

# Verify structure
ls
# Should see: src/ scripts/ config/ data/ logs/ reports/
```

### 2. Setup Environment (10 min)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn; print('✓ Ready')"
```

### 3. Configure (15 min)

```bash
# Copy example config
cp config/config.example.yaml config/config.yaml

# Edit config
nano config/config.yaml

# Update:
# - data.wrds_username: "your_username"
# - broker settings (IB or Alpaca)
# - trading.paper_trading: true (ALWAYS START WITH TRUE!)
```

### 4. Setup Data (1-2 hours)

**Option A: Use Existing Parquet (RECOMMENDED - 60 min)**
```bash
# Copy 13F data
cp -r /path/to/your/13f_parquet/* data/13f_parquet/

# Build model dataframe
python scripts/quarterly_update.py update-data
```

**Option B: Download from WRDS (SLOW - 4+ hours)**
```bash
# Download quarters (repeat for each)
python scripts/quarterly_update.py update-13f --quarter 2025-09-30

# Build dataframe
python scripts/quarterly_update.py update-data
```

### 5. Train Models (2-4 hours)

```bash
# Train all 12 models with walkforward validation
python scripts/quarterly_update.py train-models

# Verify Sharpe ~2.5-2.7
python -c "
import pandas as pd
preds = pd.read_parquet('data/models/predictions.parquet')
ensemble = preds[preds['model'] == 'Ensemble']
ic = ensemble.groupby('date_q_end').apply(lambda g: g['pred_ex'].corr(g['ret_excess_next']))
print(f'IC Sharpe: {ic.mean()/ic.std():.2f} (target: 2.5-2.7)')
"
```

### 6. Generate Signals (5 min)

```bash
# Generate trading signals
python scripts/quarterly_update.py generate-signals

# Review signals
python -c "
import pandas as pd
w = pd.read_parquet('data/signals/2025-09-30_weights.parquet')
print(f'Long: {(w[\"weight\"]>0).sum()}, Short: {(w[\"weight\"]<0).sum()}')
"
```

### 7. Test Execution (30 min)

```bash
# Dry run (no real trades)
python scripts/live_trading.py --dry-run

# Test monitoring
python scripts/monitor_daily.py

# Verify positions
python scripts/verify_positions.py
```

### 8. Paper Trade (1+ QUARTERS - MANDATORY)

```bash
# Connect to paper trading account
# IB: Start TWS with paper account, port 7497
# Alpaca: Use paper API keys

# Execute first rebalance
python scripts/live_trading.py

# Monitor daily
python scripts/monitor_daily.py
```

**⚠️ CRITICAL: Paper trade for minimum 1 quarter before going live!**

### 9. Go Live (Gradual)

After successful paper trading:
1. Month 1: Start with 20% capital
2. Month 2: Scale to 50%
3. Month 3: Scale to 75%
4. Month 4+: Full 100%

See `DEPLOYMENT_STEPS.md` for complete details.

---

## 📅 QUARTERLY WORKFLOW

### Timeline
```
Sep 30  → Q3 ends
Nov 15  → 13F data available (45 days later)
Nov 15-17 → Update data, train models
Nov 18  → Paper test
Nov 19  → LIVE execution
Nov 20 - Feb 28 → HOLD (no trading)
Mar 1   → Start Q4 cycle
```

### Commands
```bash
# Download new 13F quarter
python quarterly_update.py update-13f --quarter 2025-09-30

# Rebuild data
python quarterly_update.py update-data

# Retrain ALL 12 models
python quarterly_update.py train-models

# Generate trading signals
python quarterly_update.py generate-signals

# Review signals
cat data/signals/signals_2025Q3.csv

# Execute rebalance (market hours)
python live_trading.py
```

---

## 🏗️ SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│  QUARTERLY CYCLE (Every ~45 days after quarter end)        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────┐
         │  DATA UPDATE (30-60 min)  │
         │  - Load 13F from parquet  │
         │  - Query CRSP             │
         │  - Compute manager skill  │
         │  - Generate features      │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  FEATURE ENGINEERING      │
         │  - Interactions (15)      │
         │  - Nonlinear (18)         │
         │  - Ranks (12)             │
         │  - Cross-sectional clean  │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  REGIME DETECTION         │
         │  - S&P 500 vol & returns  │
         │  - Classify: 4 regimes    │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  MODEL TRAINING           │
         │  - Train 12 models        │
         │  - Update variance        │
         │  - Cache hyperparameters  │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  PREDICTION & ENSEMBLE    │
         │  - 3 regime models        │
         │  - Variance weights       │
         │  - Quality filters        │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  PORTFOLIO OPTIMIZATION   │
         │  - Regime parameters      │
         │  - SLSQP optimization     │
         │  - Turnover penalty       │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  EXECUTION (1-2 hours)    │
         │  - Map PERMNO → ticker    │
         │  - Generate orders        │
         │  - Submit to broker       │
         │  - Monitor fills          │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  HOLD 3 MONTHS            │
         │  - Daily monitoring       │
         │  - Risk checks            │
         │  - No intra-Q trading     │
         └───────────────────────────┘
```

---

## 🔑 KEY FEATURES

### 12-Model Ensemble
- 4 Regimes: high_vol, bear, normal, bull
- 3 Models per regime: ElasticNet, RandomForest, XGBoost
- Inverse-variance weighted ensemble
- Adapts to changing model performance

### Advanced Feature Engineering
- **Base:** Skill-weighted institutional ownership
- **Interactions:** mom×vol, mom×io, sk×mom (15 max)
- **Nonlinear:** log1p, squared terms (18 max)
- **Ranks:** Cross-sectional percentiles (12 max)
- **Cleaning:** Winsorization + robust z-score

### Regime-Adaptive Optimization
```python
high_vol: top_k=7,  max_pos=10%, risk_aversion=3.5
bear:     top_k=8,  max_pos=12%, risk_aversion=3.0
normal:   top_k=11, max_pos=16%, risk_aversion=1.5
bull:     top_k=13, max_pos=22%, risk_aversion=0.8
```

---

## ⚠️ CRITICAL REMINDERS

### 1. TRADING FREQUENCY
**THIS IS QUARTERLY TRADING, NOT DAILY/MONTHLY!**
- Rebalance once per quarter (after 13F data available)
- Hold positions for 3 months
- Monitor daily but NO intra-quarter trading
- Why? 13F data is quarterly only

### 2. START SAFELY
- ✅ Paper trading FIRST (minimum 1 quarter)
- ✅ Start with 20% of target capital
- ✅ Scale up over 3-4 quarters
- ✅ Test everything before going live

### 3. PERFORMANCE EXPECTATIONS
- **Backtest (2022-2025):** Sharpe 2.67
- **Expected Live:** Sharpe ~2.5
- **Reason for drop:** Slippage, costs, timing
- **Still excellent!** Top quartile performance

### 4. RISK MANAGEMENT
- Daily loss limit: 2%
- Drawdown limit: 15%
- Position limits: regime-specific
- Circuit breakers enabled

### 5. DATA DEPENDENCIES
- **13F:** partitioned parquet (data/13f_parquet/)
- **WRDS:** Valid subscription & credentials
- **CRSP:** msf, msenames tables
- **Broker:** IB or Alpaca account

---

## 📊 MONITORING

### Daily Checks
```bash
# Run daily monitoring
python monitor_daily.py

# Check logs
tail -f logs/strategy_*.log

# Verify positions
python verify_positions.py
```

### Performance Metrics
- Quarterly returns
- Rolling Sharpe ratio (8Q)
- Information Coefficient (IC)
- Win rate
- Turnover
- Fill quality

### Red Flags
- Sharpe < 1.5 for 2+ quarters
- IC < 0.02
- Win rate < 45% for 2 quarters
- Drawdown > 20%
- Fill rate < 80%

---

## 🆘 TROUBLESHOOTING

### Common Issues


**"WRDS connection failed"**
```bash
# Check username in config.yaml
# Test connection:
python -c "import wrds; db = wrds.Connection(wrds_username='YOUR_USERNAME'); print('OK')"
```

**"Model training fails"**
- Check src/model_training.py is complete
- Verify walkforward logic implemented
- Check logs in logs/training.log

**"No signals generated"**
- Ensure models trained successfully
- Check data/models/regime_ensemble_latest.pkl exists
- Verify latest data in model_df

**"Broker connection failed"**
- IB: Check TWS running, port correct, API enabled
- Alpaca: Check API keys in config
- Test connection before trading

---

## 📚 DOCUMENTATION

**For detailed guides:**
- 📖 **[QUICKSTART.md](QUICKSTART.md)** - Quick reference & command cheat sheet
- 📖 **[DEPLOYMENT_STEPS.md](DEPLOYMENT_STEPS.md)** - Complete step-by-step deployment
- 📖 **[docs/](docs/)** - Additional documentation

**Key commands:**
```bash
# Complete pipeline
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

## 📁 FILE REFERENCE

### Configuration
- `config.yaml` - All settings
- `requirements.txt` - Python dependencies

### Core Modules
- `src/utils.py` - Utilities
- `src/data_pipeline.py` - Data processing
- `src/feature_engineering.py` - Feature creation
- `src/regime_detection.py` - Regime classification
- `src/variance_tracker.py` - Ensemble weighting
- `src/portfolio_optimization.py` - Portfolio construction
- `src/model_training.py` - 12-model ensemble
- `src/order_execution.py` - Trading
- `src/risk_management.py` - Risk controls

### Scripts
- `quarterly_update.py` - Data/model updates
- `live_trading.py` - Execution controller
- `monitor_daily.py` - Daily monitoring
- `verify_positions.py` - Reconciliation

### Documentation
- `README.md` - This file
- `DEPLOYMENT_STEPS.md` - Detailed deployment
- `create_remaining_files.py` - File generator

---

**Recommanded Deployment Timeline:**
```
Week 1: Clone + Setup (4-8 hours)
├── Clone from GitHub
├── Install dependencies
├── Configure credentials
└── Setup data & train models

Week 1-4: Testing (30 min + ongoing)
├── Dry-run execution
├── Test all monitoring
└── Verify everything works

Months 1-3: Paper Trading (CRITICAL)
├── Execute in paper mode
├── Monitor daily
└── Validate performance

Months 4+: Live Trading (Gradual)
├── Start 20% → 50% → 75% → 100%
├── Monitor continuously
└── Scale as performance validates
```

**Next Steps:**
1. ✅ Code on GitHub (DONE)
2. 📥 Clone to trading machine
3. ⚙️ Configure & setup data
4. 🤖 Train models
5. 🧪 Test thoroughly
6. 📄 Paper trade 1+ quarters
7. 🚀 Go live gradually

**Time to first paper trade:** 1-2 days  
**Time to live trading:** 1-3 months (with testing)

See **[DEPLOYMENT_STEPS.md](DEPLOYMENT_STEPS.md)** for complete guide.


## 📊 13F Data Setup - Important!

**TWO options** - the code supports both:

### ⭐ Option A: Use Existing Parquet (RECOMMENDED)
```bash
cp -r /your/existing/13f_parquet/* data/13f_parquet/
python scripts/quarterly_update.py update-data
# Takes: ~60 minutes (FAST!)
```

**Advantages:** Much faster, saves WRDS credits, works offline

### 🌐 Option B: Download from WRDS (Fresh Start)
```bash
# If starting from scratch
python scripts/quarterly_update.py update-13f --quarter 2025-09-30
# Repeat for each quarter you need
python scripts/quarterly_update.py update-data
# Takes: 4+ hours for full history (SLOW!)
```

**Advantages:** No storage needed, always fresh

### 🔄 For New Quarters (Either Option):
```bash
# Every quarter (after 45-day filing deadline)
python scripts/quarterly_update.py update-13f --quarter 2025-12-31
python scripts/quarterly_update.py update-data
# Takes: ~5 minutes + 60 minutes
```
---
