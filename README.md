# Tennis Value Betting Model
 
**A machine learning system that detects mispriced odds in ATP tennis markets.**
 
Built a CatBoost classifier from scratch using 14 years of ATP match data (36,000+ matches), custom Elo ratings with surface specialization and inactivity decay, and an adaptive betting strategy that adjusts thresholds and position sizing by market segment.
 
Backtested on 2026 out-of-sample data: **+25.3% ROI** with the adaptive strategy, turning €1,000 into €2,914 in 3 months on 134 bets using the best odds for bigger profits. This ROI is extremely good and will soon be contrasted by LIVE-testing. 
 
<br>
 
## The Problem
 
Sports betting markets are efficient — bookmakers have teams of analysts, real-time data, and millions in liquidity correcting their lines. Beating them with public data alone is extremely hard.
 
Most ML betting models fail because they either:
- **Leak future information** into training (using market odds as features, random splits instead of temporal)
- **Overfit to noise** and collapse in production
- **Ignore execution reality** (slippage, account limitations, bankroll management)
 
This project is an honest attempt to find edge where it exists, with full awareness of where it doesn't.
 
<br>
 
## Results
 
### Model Performance (Test: Jan–Mar 2026)
 
| Metric | Model | Market | Delta |
|--------|-------|--------|-------|
| Accuracy | 70.3% | 72.7% | -2.4% |
| ROC AUC | 0.766 | 0.782 | -0.016 |
| Log Loss | 0.575 | 0.565 | +0.011 |
 
The model doesn't beat the market overall — that's expected and honest. The alpha comes from **knowing where** it has edge and betting selectively.
 
### Adaptive Strategy Backtest (2026, starting bankroll €1,000)
 
| Strategy | Bets | Final Bankroll | Return | ROI | Max Drawdown |
|----------|------|---------------|--------|-----|-------------|
| Fixed threshold (7.5%) | 249 | €1,239 | +24% | +2.7% | -24% |
| Adaptive thresholds | 179 | €2,239 | +124% | +15.6% | -30% |
| **Adaptive, no 1st Round** | **134** | **€2,914** | **+191%** | **+25.3%** | **-16%** |
 
### Where the model wins (and where it doesn't)
 
| Segment | ROI | Bets | Verdict |
|---------|-----|------|---------|
| Semifinals | +80.2% | 17 | Strong edge |
| Finals | +63.0% | 5 | Strong edge |
| 2nd Round | +23.0% | 83 | Consistent |
| Clay | +59.7% | 26 | Surface specialist Elo helps |
| 1st Round | -18.8% | 45 | Excluded from strategy |
| Masters 1000 | -6.5% | 35 | Higher threshold required |
 
The adaptive strategy encodes this knowledge: lower thresholds and larger Kelly fractions for high-alpha segments, higher thresholds or exclusion for negative segments. This means one of three things or a combination of them, the finals and semifinals are very predictable and profitable, the data from the odds on the database for finals and semifinals isn't accurate or as usually tennists in Finals and Semifinals are the ones who usually have more games throughout the year make the model predict very accurately.  
 
<br>
 
## Architecture
 
```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
│                                                         │
│  tennis-data.co.uk ──→ update_csv.py ──→ master.csv     │
│  (14 years ATP)        (daily cron)     (36k+ matches)  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                 FEATURE ENGINEERING                      │
│                                                         │
│  • Custom Elo (global + per-surface + inactivity decay) │
│  • Temporal win rates (30/60/90/180 days)               │
│  • Exponential decay weighting                          │
│  • Quality-weighted results (opponent ATP points)       │
│  • Tournament momentum (consecutive wins this week)     │
│  • H2H with temporal decay                              │
│  • Fatigue (matches/sets in recent days)                │
│  • Retirement-adjusted outcomes                         │
│  • Historical performance at tournament/location        │
│                                                         │
│  88 features total, 0 market odds (no data leakage)     │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    CATBOOST MODEL                        │
│                                                         │
│  • Temporal train/val/test split (no random leakage)    │
│  • Handles NaN natively (no imputation artifacts)       │
│  • Categorical features: surface, court, round, series  │
│  • Early stopping on held-out year                      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│               ADAPTIVE BETTING ENGINE                    │
│                                                         │
│  • Segment-specific edge thresholds                     │
│  • Fractional Kelly sizing with multipliers             │
│  • Per-segment config: round × surface × series         │
│  • Automatic exclusion of negative-alpha segments       │
│  • Real-time prediction interface with odds input       │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  DAILY SCANNER                           │
│                                                         │
│  daily_scanner.py ──→ Oddschecker scraping               │
│                   ──→ Model prediction                   │
│                   ──→ Bet/no-bet decision + Kelly sizing │
│                   ──→ Daily JSON report                  │
└─────────────────────────────────────────────────────────┘
```
 
<br>
 
## Key Design Decisions
 
**No market odds as features.** Early versions included bookmaker odds as model inputs, which inflated backtest performance to +25% ROI — but it was circular: using the market's prediction to beat the market. All reported results use only player-derived features.
 
**Elo initialized by ATP ranking.** Standard Elo starts everyone at 1500, which means the first 2-3 years of data are contaminated. The custom system initializes at `2100 - 150·log₂(rank)`, so Djokovic starts at ~2100 and a qualifier at ~1400 — reflecting real skill from day one.
 
**Retirement-adjusted outcomes.** A win by opponent retirement counts 0.75 in Elo updates and rolling win rates, because it doesn't reflect full competitive ability.
 
**Temporal decay everywhere.** Win rates, Elo (on inactivity >180 days), H2H records, and quality metrics all use exponential decay. A win 2 months ago matters more than one 2 years ago — especially relevant for injury comebacks.
 
**Segment-adaptive strategy.** Rather than a single threshold, the system uses learned configuration:
```python
SEGMENT_CONFIG = {
    "round": {
        "Semifinals": {"min_edge": 0.05, "kelly_mult": 1.5},  # aggressive
        "1st Round":  {"enabled": False},                       # excluded
    },
    "surface": {
        "Clay": {"edge_bonus": -0.02, "kelly_mult": 1.3},     # permissive
    },
    "series": {
        "Grand Slam": {"edge_bonus": -0.02, "kelly_mult": 1.3},
    },
}
# Effective threshold = round_base + surface_bonus + series_bonus
# Effective Kelly = round_mult × surface_mult × series_mult
```
 
<br>
 
## Honest Limitations
 
- **Market is still better overall.** 72.7% vs 70.3% accuracy. The edge exists only in specific segments.
- **Small test sample.** 134 bets in 3 months is not statistically conclusive. ~2,000+ bets needed for 95% confidence on a 3.5% ROI.
- **Backtest uses MaxOdds.** Assumes you always get the best price across all bookmakers. Real execution will have slippage (estimate -1-3% ROI).
- **Account limitations.** Bookmakers limit or ban profitable bettors. Expected account lifetime: 2-8 weeks at Bet365-tier houses.
- **No CLV tracking.** The true test of a model is whether it consistently beats the closing line. This requires live tracking, which backtest data doesn't provide.
 
<br>
 
## Usage
 
### Setup
 
```bash
pip install catboost pandas numpy scikit-learn matplotlib requests openpyxl xlrd
```
 
### Train the model
 
```bash
# Place your CSV/Excel files from tennis-data.co.uk in raw_data/
python tennis_v4.py
```
 
### Daily workflow
 
```bash
# 1. Update data (run each morning)
python update_csv.py
 
# 2. Scan today's matches
python daily_scanner.py --tournament=atp-miami --bankroll=1000
 
# Or input matches manually
python daily_scanner.py --manual --bankroll=1000
```
 
### Predict a single match
 
```python
predict_match(
    "Sinner J.", "Alcaraz C.",
    surface="Clay", court="Outdoor",
    round_name="Semifinals", series="Grand Slam",
    p1_odds=2.40, p2_odds=1.60,
    bankroll=1000
)
```
 
Output:
```
  Modelo: Sinner 28.7% | Alcaraz 71.3%
  Estrategia: Threshold 2.0% | Kelly x2.5
 
  ✅ APOSTAR: Alcaraz C. @ 1.60
     Edge: +8.8% | EV/€: +0.140
     → APOSTAR: 30.00€ de 1000€
 
  ❌ Sinner J. @ 2.40
     Edge -12.9% < threshold 2.0%
```
 
<br>
 
## Project Structure
 
```
├── tennis_v4.py          # Full pipeline: data → features → model → prediction
├── update_csv.py         # Daily CSV updater (tennis-data.co.uk)
├── daily_scanner.py      # Match scanner with Oddschecker scraping
├── tennis_v4.cbm         # Trained CatBoost model
├── tennis_master.csv     # Master dataset
├── raw_data/             # Original CSV/Excel files (2012-2026)
└── reports/              # Daily JSON reports
```
 
<br>
 
## Data
 
All match data from [tennis-data.co.uk](http://www.tennis-data.co.uk/) (2012–2026). Features are computed exclusively from match results, rankings, and tournament metadata — no proprietary or paid data sources.
 
<br>
 
## Tech Stack
 
Python · CatBoost · pandas · NumPy · scikit-learn · matplotlib · requests
 
<br>
