# NBA Draft Prediction Model

An XGBoost model that ranks NBA draft prospects using only their college box scores. It is trained to predict career NBA impact (DARKO, an on/off-aware impact metric) and judged on career VORP, then compared against the picks NBA front offices actually made. A custom hill-climbing optimizer tunes its features and settings, scoring every candidate draft by draft with walk-forward validation.

**[→ View the full results](https://sdm-commits.github.io/nba-model/)**

---

## Results

For each NBA draft from 2015 to 2022, the model ranks that year's college first-round picks and keeps its top 10, using only stats from earlier drafts. Their career VORP is compared against the actual top 10 picks, which NBA teams chose from college, international, and G League pools.

| Metric | Model | NBA GMs |
|---|---|---|
| Career VORP from top-10 picks | **631** | 483 |
| Drafts won | **7 of 8** | — |
| Bust rate (negative career VORP) | **15%** | 31% |
| Productive players found outside the real top 10 | **28** | — |

The model's picks produced **30% more career value** across 80 selections. The board averages three training runs with different random seeds. Career stats are as of November 13, 2026 (2026–27 season in progress).

VORP is built from box-score stats, the same kind of numbers the model reads, so the same boards were also graded on two impact metrics that use on/off data:

| Graded on | NBA GMs | Original model (VORP-trained) | Current model (DARKO-trained) |
|---|---|---|---|
| Career VORP | 483 | 601 (+24%, won 5/8) | **631 (+30%, won 7/8)** |
| DARKO over replacement (through 2025–26) | 1105 | 1164 (+5%) | **1242 (+12%)** |
| xRAPM over replacement | 1047 | 1156 (+10%) | **1215 (+16%)** |

Settings were tuned on these same eight drafts, so totals are somewhat optimistic. With its starting settings and no tuning, the current model scores 600 (+24%) and still wins 7 of 8.

xRAPM includes the 2026–27 season so far, weighted by games played out of 82. DARKO runs through 2025–26 because its 2026–27 numbers aren't published yet.

### Players the model found early

Ranked in the model's top 10 but drafted outside the top 10 (career VORP):

| Player | Draft | Actual pick | Career VORP |
|---|---|---|---|
| Shai Gilgeous-Alexander | 2018 | 11 | 37.7 |
| Donovan Mitchell | 2017 | 13 | 29.7 |
| Domantas Sabonis | 2016 | 11 | 27.8 |
| Pascal Siakam | 2016 | 27 | 21.4 |
| Tyrese Haliburton | 2020 | 12 | 19.2 |
| Myles Turner | 2015 | 11 | 13.3 |
| Josh Hart | 2017 | 30 | 12.8 |
| Delon Wright | 2015 | 20 | 11.1 |

### Blending with scouts

Mixing in pre-draft scout consensus barely moves the model. Blended rank = w × model rank + (1 − w) × consensus rank, within each year's college first-round picks (players missing from the consensus board get rank 40):

| Board | Career VORP, top 10s 2015–2022 |
|---|---|
| Model + scouts (89% model) | **635** |
| Model only | 631 |
| Scout consensus only | 487 |
| NBA GMs | 483 |

The best blend changes one pick in eight drafts (2020: Saddiq Bey in place of Udoka Azubuike), and any blend under about 80% model weight does worse than the model alone. Consensus boards: Rookie Scale (2021–22), Hoops Prospects (2019–20), and averages of final public big boards from SI, CBS Sports, Bleacher Report, ESPN/DraftExpress, Yahoo and the Washington Post (2015–18). The weight is chosen on the same eight drafts, so the blend's edge is an upper bound.

### 2026 draft

The page shows the model's ranking of the 2026 college first-rounders next to where they were actually picked. Its top five: Brayden Burries (pick 10), Aday Mara (12), Kingston Flemings (8), Yaxel Lendeborg (11) and Cameron Boozer (3). Boards for every draft from 2015 to 2026 are on the [results page](https://sdm-commits.github.io/nba-model/).

## How It Works

### Data & Features

College stats sourced from [Barttorvik](https://barttorvik.com/) (2010–2026), supplemented with international basketball datasets from Kaggle. The model uses **150 engineered features** spanning:

- Shooting, creation, and finishing profiles
- Defensive composites (stocks, stops, adjusted defensive efficiency)
- Trajectory features (BPM slope, acceleration, consistency across seasons)
- Team context (Barthag, SOS, tournament seeding, teammate-relative metrics)
- Positional archetypes inspired by [EuroLeague clustering research](https://sltsportsanalytics.substack.com/p/decoding-euroleague-positions-a-data) (three-and-D, stretch big, point forward, rim runner, combo creator)
- Recruit score interactions and conference strength adjustments

The "BPM" shown on the page is Torvik's GBPM, which is what the dataset's BPM columns actually contain.

The model learns to predict career NBA impact: each season's [DARKO](https://apanalytics.shinyapps.io/DARKO/) rating above replacement level (−2.0), summed over the career and log-scaled. It is judged on career VORP. Upperclassmen are discounted (juniors 6%, seniors 18%), except seniors who were elite college producers (BPM ≥ 12). (The original version was trained directly on VORP.)

### Position System

Continuous probabilistic classification — each player gets `(is_guard, is_wing, is_big)` scores between 0 and 1 based on height sigmoid curves blended with Torvik role labels. Post-classification reclassification detects point forwards and stretch bigs, shifting their positional weight from big to wing to prevent penalization through big shrinkage adjustments.

### Validation

Strict walk-forward: for each draft year, the model trains only on prior years. No future data ever enters a prediction. Each configuration is scored on all eight test drafts (2015–2022), averaged over three training runs with different random seeds, with floating-point math pinned so results are identical on any CPU.

### Autonomous Optimizer

A hill-climbing loop that runs the full training pipeline over and over. Each iteration:

1. **Mutates** one to three settings: feature on/off switches, thresholds, position adjustments, or XGBoost parameters
2. **Trains** the model draft by draft (walk-forward), three times with different seeds
3. **Scores** the result on the same yardstick as the results: total career VORP of the model's top 10, 2015–2022
4. **Keeps** the change only if it adds at least 2 career VORP and helps in at least as many drafts as it hurts

The current model ran 418 iterations; 5 changes survived.

## Tech Stack

- **Python** — pandas, NumPy, SciPy
- **XGBoost** — gradient-boosted regression
- **DARKO / xRAPM** — NBA impact metrics (training target and secondary grading)
- **Barttorvik** — college advanced stats (2010–2026)
- **Kaggle** — international basketball datasets
- **Basketball Reference** — NBA career VORP labels
- **Walk-forward CV** — temporal cross-validation with no data leakage
- **Hill-climbing optimizer** — custom mutation loop, seed-averaged fitness

## Known Limitations

- No use of film, interviews, big boards, or medicals
- Cannot evaluate character, coachability, injury risk, or team fit
- Only evaluates college players — competing against GM boards that also include international and G League prospects
- Settings were tuned on the same drafts used to report results
- VORP is an imperfect measure of NBA value (doesn't capture defense well, penalizes players on bad teams)
- Small test set per year (~25 college first-rounders) means single-draft results are noisy

## License

© 2025 Scott Middlecamp. All rights reserved.
