# NBA Draft Prediction Model

An XGBoost-based model that predicts NBA career value (VORP) from pre-draft NCAA statistics, supplemented with international basketball datasets. Trained and optimized using a custom evolutionary algorithm that autonomously discovers feature configurations, position calibration parameters, and regularization settings through walk-forward cross-validation.

**[→ View the full results](https://sdm-commits.github.io/nba-model/)**

---

## Results

Evaluated against actual NBA drafts from 2015–2022. The model sees only college box scores — no film, interviews, medicals, or international/G League players. It competes against GM boards that draw from the entire global talent pool.

| Metric | Model | NBA GMs |
|---|---|---|
| Career VORP from top-10 picks | **550** | 446.9 |
| Bust rate (negative career VORP) | **18%** | 30% |
| Productive players found outside real top 10 | **26** | — |

The model generates **23% more career value per pick** across 80 selections.

## How It Works

### Data & Features

College stats sourced from [Barttorvik](https://barttorvik.com/) (2010–2026), supplemented with international basketball datasets from Kaggle. The model uses **144 engineered features** spanning:

- Shooting, creation, and finishing profiles
- Defensive composites (stocks, stops, adjusted defensive efficiency)
- Trajectory features (BPM slope, acceleration, consistency across seasons)
- Team context (Barthag, SOS, tournament seeding, teammate-relative metrics)
- Positional archetypes inspired by [EuroLeague clustering research](https://sltsportsanalytics.substack.com/p/decoding-euroleague-positions-a-data) (three-and-D, stretch big, point forward, rim runner, combo creator)
- Recruit score interactions and conference strength adjustments

Target variable is 3-year cumulative NBA VORP with a log-shifted transformation and experience-weighted sampling.

### Position System

Continuous probabilistic classification — each player gets `(is_guard, is_wing, is_big)` scores between 0 and 1 based on height sigmoid curves blended with Torvik role labels. Post-classification reclassification detects point forwards and stretch bigs, shifting their positional weight from big to wing to prevent penalization through big shrinkage adjustments.

### Validation

Strict walk-forward: for each draft year, the model trains only on prior years. No future data ever enters a prediction. Each configuration is evaluated against real career VORP outcomes across all test years (2015–2022).

### Autonomous Optimizer

A custom evolutionary algorithm that runs the full training pipeline in a loop. Each generation:

1. **Mutates** the configuration — feature inclusion/exclusion (from 200+ candidates), XGBoost hyperparameters, position calibration thresholds, conference strength multipliers, scoring cutoffs, post-prediction position adjustments, and senior discount parameters
2. **Trains** an XGBoost model using walk-forward CV
3. **Scores** the result with a weighted fitness function (overall correlation, clean-year correlation, top-10 overlap, per-position correlations, bust penalty)
4. **Selects** — improvements survive, regressions are discarded

The optimizer maintains a population of top configurations and uses multiple mutation strategies: focused mutations on high-correlation parameters, position-specific tuning, XGBoost hyperparameter sweeps, crossover between top candidates, and random restarts to escape local optima. Mutation probabilities are weighted by historical feature-to-VORP correlations and adapt based on which parameters have historically produced improvements.

Each model converges in 150–200 autonomous generations. Across nine model iterations, the fitness score improved from ~0.25 to 0.65+. The final 144-feature configuration was discovered by the algorithm, not hand-selected.

## Tech Stack

- **Python** — pandas, NumPy, SciPy
- **XGBoost** — gradient-boosted regression with experience-weighted sampling
- **Barttorvik** — college advanced stats (2010–2026)
- **Kaggle** — international basketball datasets
- **Basketball Reference** — NBA career VORP labels
- **Walk-forward CV** — temporal cross-validation with no data leakage
- **Evolutionary optimizer** — custom mutation engine with adaptive strategy selection

## Known Limitations

- No use of film, interviews, big boards, or medicals
- Cannot evaluate character, coachability, injury risk, or team fit
- Only evaluates college players — competing against GM boards that also include international and G League prospects
- VORP is an imperfect measure of NBA value (doesn't capture defense well, penalizes players on bad teams)
- Small test set per year (~10-30 drafted college players) means individual year correlations are noisy

## License

© 2025 Scott Middlecamp. All rights reserved.
