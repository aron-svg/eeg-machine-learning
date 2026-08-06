# EEG Machine Learning — Emotion Classification

A machine learning pipeline that predicts a subject's **arousal**, **valence**, and **dominance** (the SAM/AVD model) from precomputed EEG features, using nested, group-aware cross-validation to avoid data leakage between correlated recording windows.

## Task

For each EEG recording session, feature windows (Hjorth parameters, differential entropy, hemispheric asymmetry, ...) are extracted upstream and stored per subject. Each window also carries the subject's self-reported arousal/valence/dominance rating (1-9 scale) for the video stimulus shown during that window.

The pipeline:
1. Binarizes each rating at a threshold (`label >= 5` → "high", else "low").
2. Trains and cross-validates a set of candidate classifiers per target.
3. Reports the best model per target with its held-out performance.

**Current scope: intra-subject** (one subject's file at a time). Multi-subject leave-one-subject-out (LOSO) evaluation is the next phase — the data loader already globs `*_features.npz`, so extending it to concatenate several subjects is the main remaining step.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates `.venv/` and installs everything pinned in `pyproject.toml` (numpy, pandas, scikit-learn, scipy, plus black/flake8 for linting).

## Input data

Place one `.npz` file per subject in `data/input/` (e.g. `HZO024_features.npz`, matched by `NPZ_GLOB_PATTERN` in `config.py`). Each archive must contain:

| key | shape | description |
|---|---|---|
| `X` | `(n_windows, n_features)` | precomputed feature matrix |
| `y` | `(n_windows, 3)` | arousal, valence, dominance ratings (1-9) |
| `trial_index` | `(n_windows,)` | which trial/video each window belongs to — used to group cross-validation folds so windows from the same trial never leak between train and test |
| `subject_id`, `window_start`, `window_end`, `media_filename` | — | carried along, not used by the current pipeline |

`data/input/` and `data/output/` are gitignored — this repo doesn't version raw EEG data or generated results.

## Running the pipeline

```bash
cd src
python main.py
```

This loads the subject file, runs cross-validation for every (target × active model) combination in parallel, logs the best model per target, and writes `data/output/classification_results.json`.

## Configuring a run

Everything tunable lives in `src/config.py` — no magic numbers elsewhere in the codebase:

- **`BINARIZATION_THRESHOLD`** — where the 1-9 rating splits into "low"/"high".
- **`N_SPLITS` / `INNER_CV_SPLITS`** — outer (evaluation) and inner (hyperparameter search) fold counts for the grouped `StratifiedGroupKFold`.
- **`MODEL_PARAM_GRIDS`** — the full catalog of 8 candidate models and their hyperparameter grids: `logistic_elasticnet`, `random_forest`, `extra_trees`, `gradient_boosting`, `svm_rbf`, `knn`, `lda`, `mlp`.
- **`ACTIVE_MODELS`** — which of the 8 actually run this time. Each `(target, model)` pair gets its own CPU core (see below), so keep `len(TARGET_NAMES) * len(ACTIVE_MODELS)` comfortably under your machine's core count.

## Methodology notes

- **Grouped CV**: windows from the same trial share the same stimulus and are temporally close, so a random train/test split would leak information. Every split — outer evaluation and inner hyperparameter search — is grouped by `trial_index` via `StratifiedGroupKFold`.
- **No leakage from scaling/tuning**: `StandardScaler`, feature selection, and model fitting all happen inside each training fold via an `sklearn.Pipeline`, never on the full dataset.
- **Feature selection**: models with embedded selection (L1/ElasticNet sparsity, tree impurity/gain) use all 515 features directly. Models without one (`svm_rbf`, `knn`, `lda`, `mlp`) get an explicit `SelectKBest` step, with `k` tuned like any other hyperparameter.
- **Parallelism**: `tools/cv.py` runs every `(target, model)` pair as its own worker process (`joblib.Parallel`), while each individual `GridSearchCV` stays single-core (`N_JOBS_PER_MODEL`) to avoid oversubscribing the machine.

## Project structure

```
src/
├── main.py              # entry point: checks input, calls machine_learning.main_process()
├── config.py            # all tunable constants (paths, targets, CV, model grids, ACTIVE_MODELS)
├── data_loader.py        # loads a subject's .npz feature archive
├── machine_learning.py   # orchestrates: load -> binarize -> cross-validate -> save results
├── tools/                 # ML building blocks
│   ├── preprocessing.py   # label binarization, target selection
│   ├── models.py          # pipeline + estimator factory for the 8 candidate models
│   ├── cv.py               # nested, grouped, parallel cross-validation
│   └── metrics.py          # score summarization, best-model selection, JSON output
├── logger_init.py / logging_config.py / logger_config.yaml   # logging setup
data/
├── input/                # per-subject .npz feature files (gitignored)
└── output/               # classification_results.json (gitignored)
```

## Code style

Formatted with `black` (line length 79) and linted with `flake8`; import order enforced by `isort` — both configured in `pyproject.toml`. No source file exceeds 200 lines by convention, to keep each module focused on one responsibility.

## Docker

```bash
docker-compose up
```

Builds the image and runs `python src/main.py` once against whatever is mounted in `data/`. There's no HTTP server in this pipeline — the exposed port in `docker-compose.yml` is inherited from the original template and isn't currently used.
