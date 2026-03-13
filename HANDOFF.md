# Alternative Strategies: NLP, Satellite, and Flow Data

## Project Handoff from strat-testing

This document specifies the alternative-data strategies project. It is written for a developer picking up where the `strat-testing` backtesting sandbox left off. Read this before writing any code.

---

## Why this project exists

The `strat-testing` project tested ~20 strategy variants across 20 years of data on public price data. The conclusions:

1. **No alpha was found on public price features.** Every alpha p-value across every strategy was insignificant at 5%. The ML return-prediction model (ridge regression, gradient boosting) did not beat a simple 12-1 momentum rule on cross-period consistency (0.05 Sharpe spread for momentum vs 0.53 for ML Ridge).

2. **The features-to-returns mapping is non-stationary.** ML models trained on 2010-2019 data optimized for relationships that no longer held by 2023. Simple heuristics (trailing return) avoid this by not estimating conditional relationships at all.

3. **What survived:** well-known risk premia (cross-sectional momentum, trend following) on liquid ETFs, harvested cheaply. A 50/50 blend of XS Momentum + Trend Following delivered 0.60-0.84 Sharpe across three periods (2005-2013, 2014-2019, 2020-2026) with ~$1K/year in costs. This is not alpha -- it's diversified beta.

4. **The lesson for ML:** the bottleneck is not model complexity or feature engineering on price data. It's that price data is already priced. The path to alpha requires information the market hasn't fully incorporated: alternative data, shorter horizons, or proprietary signals.

This project pursues that path.

---

## What we're building

Three alternative-data signal pipelines, each targeting a different information edge:

### Signal 1: NLP on SEC Filings (10-K, 10-Q, 8-K)

**Thesis:** The language in corporate filings contains information about future fundamentals that isn't fully reflected in the stock price at the time of filing. Changes in tone, risk factor disclosures, and management discussion language predict future returns and volatility.

**Why this might work:**
- Filings are legally required disclosures. Companies must describe risks, uncertainties, and material changes. The language is constrained -- they can't just say nothing.
- The market reacts to filings, but the reaction is often slow for non-headline information buried deep in the document (the MD&A section, risk factors, footnotes).
- Academic evidence: Loughran & McDonald (2011) showed that filing sentiment predicts returns. Subsequent work (Jha & Liu, 2024; many others) confirmed this with modern NLP.
- The signal decays over days to weeks. This is a medium-frequency signal (weekly to monthly rebalancing).

**Data sources:**
- SEC EDGAR full-text filings: free, comprehensive, goes back to 1993. API at `https://efts.sec.gov/LATEST/search-index?q=...` and bulk downloads at `https://www.sec.gov/cgi-bin/browse-edgar`.
- Filing dates and metadata: EDGAR index files (`https://www.sec.gov/Archives/edgar/full-index/`).
- SEC rate limits: 10 requests/second with a `User-Agent` header identifying your app.

**Feature extraction pipeline:**
1. Download raw filing HTML/text from EDGAR for the target universe.
2. Parse and extract sections: MD&A (Item 7), Risk Factors (Item 1A), Financial Statements (Item 8).
3. Compute features:
   - **Sentiment scores**: Loughran-McDonald financial sentiment dictionary (purpose-built for financial text; do NOT use VADER or general-purpose sentiment -- they miscategorize financial terms like "liability" and "tax").
   - **Sentiment change**: delta in sentiment score vs the prior filing for the same company. A deteriorating tone from one quarter to the next is more predictive than the absolute level.
   - **Readability/complexity**: Gunning Fog index, document length, proportion of complex sentences. More obfuscated filings correlate with worse future outcomes (Li, 2008).
   - **Embedding similarity**: cosine similarity between current filing and prior filing using a sentence transformer. Large changes in embedding space indicate material content changes even when keyword-level sentiment is flat.
   - **Named entity extraction**: new risk factors, new litigation mentions, new counterparty names. These are event-like signals.
4. Align features to filing dates (point-in-time: you can only use the filing after it's filed, not before).

**Model:**
- Predict next-quarter excess return (stock return minus sector ETF return) from filing features + basic price features (momentum, vol).
- Ridge regression first. If that works, try gradient boosting.
- Walk-forward: train on all filings up to date T, predict post-filing returns for filings at date T.

**Validation approach:**
- Event study: compute cumulative abnormal return (CAR) in the 1-5 days and 5-20 days after filing. If the NLP signal has predictive power, the CAR should be monotonically increasing in quintiles of the signal (top quintile has highest CAR, bottom has lowest).
- Long-short portfolio: go long top-quintile sentiment, short bottom-quintile. Monthly rebalance. Measure Sharpe, alpha vs Fama-French 3-factor or 5-factor model.
- Out-of-sample: train on 2010-2017, test on 2018-2021, validate on 2022-2025. Do NOT tune on the test set.

**Libraries:**
- `edgar` or `sec-edgar-downloader` for EDGAR access
- `beautifulsoup4` for HTML parsing (already in the venv)
- `loughran-mcdonald` or manual dictionary load from [Loughran-McDonald master dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
- `sentence-transformers` for embedding-based features
- `spacy` for NER and linguistic features

---

### Signal 2: Satellite / Geospatial Data

**Thesis:** Physical-world measurements (parking lot occupancy, crude oil storage tank levels, shipping activity, nighttime light intensity) provide leading indicators of economic activity and corporate revenue before quarterly earnings are reported.

**Why this might work:**
- Quarterly earnings are announced 4-6 weeks after quarter-end. During that 4-6 week window, the market is guessing. Physical measurements during the quarter provide ground truth.
- The classic example: RS Metrics (now Orbital Insight) counted cars in Walmart parking lots via satellite and predicted same-store sales. The correlation between parking lot counts and revenue was >0.9.
- This is the most capital-intensive signal source. Raw satellite imagery is expensive ($1-50 per square kilometer per pass). Processed datasets (aggregated, anonymized) are cheaper but still $10K-100K/year for useful coverage.

**Practical approach for this project (free/cheap data):**
- We are NOT buying satellite imagery. Instead, we use publicly available geospatial proxies:
  - **NOAA nighttime lights** (VIIRS Day/Night Band): free, global, measures economic activity via luminosity. Monthly composites available from [NOAA/NCEI](https://www.ngdc.noaa.gov/eog/viirs/download_dnb_composites.html). Resolution is ~750m -- enough to measure city-level or country-level economic activity but not individual store parking lots.
  - **AIS shipping data** (Automatic Identification System): vessel tracking data. Free from [MarineTraffic](https://www.marinetraffic.com/) (limited) or [UN Global Platform](https://unglobalpulse.org/). Port congestion and shipping volume correlate with trade activity and commodity demand.
  - **Google Trends**: free, daily, measures search interest for economic keywords (job searches, luxury goods, real estate). Choi & Varian (2012) showed Google Trends predicts short-term economic indicators.
  - **FRED economic data**: free API for macro indicators (initial claims, ISM PMI, retail sales). These are monthly/weekly with known release dates. The signal is in the surprise (actual vs consensus), not the level.

**Feature extraction pipeline:**
1. Download VIIRS nighttime light composites for target regions (US, China, Europe).
2. Compute month-over-month luminosity change for each region. Aggregate to country level.
3. Correlate with regional ETF returns (EFA, EEM, FXI, etc.) at a 1-3 month lag.
4. For Google Trends: compute z-score of search volume for economic keywords (e.g., "unemployment," "bankruptcy," "new car") relative to 52-week history.
5. For FRED: compute surprise (actual release minus consensus from surveys).

**Model:**
- Predict next-month country/sector ETF returns from geospatial + macro features.
- This is a macro-level signal, not stock-level. Use it for country/sector rotation.
- The universe is the same ETF set from the factor research (EQUITY_ETFS + MULTI_ASSET).

**Validation approach:**
- Granger causality test: does the satellite/Google feature Granger-cause ETF returns at 1-3 month lags? This is a necessary condition for the signal to be useful.
- Incremental R-squared: add the satellite features to the existing ridge model from the ML research. Does out-of-sample R-squared improve?
- Walk-forward backtest using the same engine from strat-testing. Compare to XS Momentum baseline (Sharpe 0.69-0.74).

---

### Signal 3: Order Flow / Positioning Data

**Thesis:** Large institutional trades create temporary price pressure. The direction and magnitude of order flow predicts short-term returns (hours to days). Aggregate positioning data reveals crowded trades that are vulnerable to unwinds.

**Why this might work:**
- Kyle (1985): informed traders move prices gradually through their order flow. The flow itself is informative.
- CFTC Commitments of Traders (COT): weekly report showing futures positioning by commercials, large speculators, and small speculators. Extreme positioning (everyone on one side) historically precedes reversals.
- Options flow: unusual options activity (large block trades, put/call ratio spikes) contains information about short-term directional bets by informed participants.
- ETF flow: daily creation/redemption data shows institutional money entering and leaving sectors. Extreme inflows are contrarian negative (herding), extreme outflows are contrarian positive.

**Data sources:**
- **CFTC COT reports**: free, weekly, from [cftc.gov](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm). Covers all major futures (equity index, Treasury, commodity, currency).
- **ETF fund flows**: daily, from ETF issuers (iShares, SPDR, Vanguard publish AUM which implies flows) or aggregators. `yfinance` provides some of this via Ticker.get_shares_full() but coverage is spotty.
- **Options data**: CBOE provides aggregate statistics (put/call ratio, total volume, VIX term structure). Detailed options flow requires a paid data vendor (OptionMetrics, LiveVol).
- **Short interest**: FINRA publishes bi-monthly short interest for NYSE/NASDAQ stocks. Free, 2-week delay.

**Feature extraction pipeline:**
1. Download COT reports. Compute net positioning for large speculators in equity index futures (ES, NQ), Treasury futures (TY, US), and commodity futures (CL, GC).
2. Compute z-score of net positioning vs 52-week history. Extreme readings (>2 sigma) are the signal.
3. Compute put/call ratio for SPX options (from CBOE). Z-score vs 52-week.
4. Compute ETF flow momentum: 5-day vs 20-day average daily flow for sector ETFs.
5. All features must be lagged by the publication delay (COT: Tuesday data published Friday; options: T+1; flows: T+0 to T+1).

**Model:**
- Short-term return prediction (1-5 day horizon) for equity index and sector ETFs.
- This is higher-frequency than the filing signal. Rebalance weekly.
- The signal is primarily contrarian: extreme bullish positioning -> expect mean reversion.

**Validation approach:**
- Quintile analysis: sort weekly COT readings into quintiles. Compute average next-week return per quintile. If the signal works, the relationship should be monotonic.
- Conditional analysis: does the flow signal add value above momentum? Run double-sort: first sort by momentum, then within each momentum quintile sort by flow. If flow adds information, the spread within momentum quintiles should be significant.
- Walk-forward backtest using the engine. Weekly rebalance, 3bps costs. Compare to XS Momentum.

---

## Architecture

### Shared infrastructure (from strat-testing)

The backtesting engine, metrics, and analysis tools from `strat-testing` are production-quality and should be reused:

- **Engine** (`src/backtest/engine.py`): share-count tracking, transaction costs, rebalance threshold, `None` vs `{}` semantics, benchmark in lookback. Copy this wholesale.
- **Metrics** (`src/backtest/metrics.py`): Sharpe, Sortino, Calmar, alpha/beta, rolling beta. Copy.
- **Alpha assessment** (`src/analysis/alpha.py`): t-test, bootstrap CI, rolling alpha, IC. Copy.
- **Price fetcher** (`src/data/prices.py`): yfinance with parquet cache, dividend-adjusted, SSL session. Copy.
- **Feature engineering patterns** (`src/ml/features.py`): the per-ETF and market-level feature computation. The pipeline pattern (compute features at date T from lookback data, no look-ahead) is the right template for alternative data features.

### New infrastructure needed

```
src/
  data/
    edgar.py          # SEC EDGAR filing downloader and parser
    cot.py            # CFTC Commitments of Traders report fetcher
    fred.py           # FRED economic data API client
    google_trends.py  # Google Trends fetcher
    satellite.py      # VIIRS nighttime light data processor
  nlp/
    sentiment.py      # Loughran-McDonald dictionary sentiment
    embeddings.py     # Filing embedding computation
    filing_parser.py  # 10-K/10-Q/8-K section extraction
  signals/
    filing_signal.py  # NLP filing signal: features -> prediction -> weights
    geo_signal.py     # Satellite/macro signal
    flow_signal.py    # Order flow / positioning signal
  strategies/
    nlp_strategy.py   # Strategy wrapper for filing signal
    geo_strategy.py   # Strategy wrapper for satellite signal
    flow_strategy.py  # Strategy wrapper for flow signal
```

### Data storage

Alternative data is larger than price data. Use:
- Parquet for tabular features (same as strat-testing)
- Raw filings: store as gzipped text in `data/filings/{ticker}/{filing_date}.txt.gz`
- Satellite composites: store as GeoTIFF or CSV aggregates in `data/satellite/`
- COT reports: CSV, one file per week in `data/cot/`

---

## Validation framework

Every signal must pass these gates before it's considered tradeable:

### Gate 1: Is the feature predictive in isolation?

- Compute the rank IC (Spearman correlation) between the feature and next-period returns across all assets and all dates.
- IC > 0.02 (2%) is meaningful for monthly returns. IC > 0.05 is strong.
- Use the `information_coefficient()` function from `src/analysis/alpha.py` (already implemented, raises on insufficient data).

### Gate 2: Is the signal monotonic in quintiles?

- Sort the universe into 5 quintiles by the signal each period.
- Compute average next-period return per quintile.
- The relationship should be monotonic (Q1 < Q2 < Q3 < Q4 < Q5 for a long signal). Non-monotonicity suggests the signal is noisy or nonlinear.
- This is the standard factor portfolio test from the academic literature.

### Gate 3: Does it survive walk-forward backtesting?

- Build a strategy that goes long the top quintile (and optionally short the bottom).
- Run through the engine with realistic costs.
- Measure Sharpe, max drawdown, and number of trades.
- Compare to XS Momentum baseline (Sharpe 0.69-0.74 across periods, the bar from strat-testing).

### Gate 4: Is it robust across time periods?

- Test on at least two non-overlapping periods of 3+ years each.
- The strat-testing project used three periods (2005-2013, 2014-2019, 2020-2026). The Sharpe spread across periods should be < 0.3.
- If the signal only works in one period, it's overfit to that regime.

### Gate 5: Does it add value above existing signals?

- Run a combined model: existing features (momentum, vol) + new alternative features.
- Compute the incremental R-squared from the alternative features.
- If incremental R-squared is < 0.5%, the signal doesn't add enough above what's already available from price data.
- Run a double-sort: sort by momentum, then within each momentum quintile sort by the alternative signal. If the within-quintile spread is insignificant, the signal is redundant.

### Anti-patterns to avoid (from strat-testing experience)

1. **Don't use today's data to backtest the past.** The analyst-rating strategy in strat-testing used March 2026 consensus ratings applied to January 2020 trades. The fix was reconstructing point-in-time consensus from individual upgrade/downgrade events with exact dates. Apply the same principle to filings (use filing date, not period-end date) and COT (use publication date, not report date).

2. **Don't silently swallow errors.** Every `try/except` in the original strat-testing codebase was replaced with explicit raises. If a filing fails to parse, raise. If EDGAR returns an error, raise. Silent failures corrupt backtest results.

3. **Watch for the `{}` vs `None` distinction.** The engine treats `{}` as "go to cash" and `None` as "keep existing positions." Intermittent signals (NLP fires only when there's a new filing) should return `None` between events to hold positions, not `{}` which liquidates.

4. **Transaction costs matter at higher frequencies.** The flow signal targets weekly rebalancing. At 10bps per trade on individual stocks, weekly turnover destroyed every stock-level strategy in strat-testing. Use ETFs (3bps) or model costs explicitly. For single stocks, budget 5-15bps depending on market cap and urgency.

5. **Overfitting is the central enemy.** The ML ridge model got 0.91 Sharpe in one period and 0.38 in another. With thousands of NLP features on a few hundred filings per quarter, overfitting will be even more severe. Regularize aggressively. Prefer ridge over gradient boosting until you have >10,000 training observations.

---

## Priority order

1. **NLP on filings** -- start here. The data is free (EDGAR), the academic evidence is strong, and the infrastructure is pure Python (no GPU needed for dictionary-based sentiment; embeddings can run on CPU with distilled models). This signal has the clearest path to a testable hypothesis within a week.

2. **Order flow** -- second. COT data is free and well-structured. The weekly frequency aligns with the engine's existing capabilities. The contrarian positioning signal is well-documented in the practitioner literature.

3. **Satellite/geospatial** -- last. The free data (VIIRS, Google Trends) is coarse and the signal is macro-level (country/sector, not stock-level). This is the hardest to get signal-to-noise from but the most interesting conceptually.

---

## What success looks like

- A signal with IC > 0.03 that is monotonic in quintiles and survives walk-forward backtesting across two periods.
- A strategy with Sharpe > 0.5 that adds incremental value (> 0.5% R-squared) above the XS Momentum baseline.
- A clear understanding of WHY the signal works (what information it captures that price data doesn't) and WHEN it fails (which regimes degrade it).

The strat-testing project proved that finding alpha is hard. This project should prove whether alternative data makes it easier -- and if so, how much.
