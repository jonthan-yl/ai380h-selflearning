# Early Warning for Sepsis in the ICU — a self-learning tutorial

Predict, hour by hour, whether an ICU patient will meet Sepsis-3 criteria within the next ~6 hours, using the
open-access **PhysioNet/CinC Challenge 2019** dataset (40,336 ICU patients from two hospital systems).

The tutorial walks through the full workflow and compares four approaches:

| # | Approach | Type |
|---|---|---|
| 1 | SIRS bedside rule | clinical baseline |
| 2 | Logistic regression | ML |
| 3 | LightGBM gradient-boosted trees + SHAP explanations | ML |
| 4 | GRU recurrent neural network on raw hourly sequences | DL |

It also covers patient-level data splitting, causal feature engineering (with an automated leakage test),
patient-level alert metrics (sensitivity, specificity, PPV, lead time), calibration, and **external validation**
(train at Hospital A, test unchanged at Hospital B).

## Files

| File | What it is |
|---|---|
| `sepsis_early_warning.ipynb` | **The tutorial** — run top to bottom (outputs included) |
| `src/sepsis_early_warning.py` | Same content as a plain Python script (percent-cell format), handy for diffs |
| `src/build_notebook.py` | Rebuilds the notebook from the script (`--execute` also runs it) |
| `requirements.txt` | Python dependencies |

## How to run

**Google Colab (easiest):** upload `sepsis_early_warning.ipynb`, then *Runtime → Run all*. All packages are
pre-installed on Colab.

**Locally:**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab sepsis_early_warning.ipynb
```

The first run downloads ~40k small files (~1 GB on disk with the cache) into `data/`; later runs reuse the cache.
Set `FAST_MODE = True` in the configuration cell for a quick run on 5,000 patients per hospital.

## Data

Reyna MA, *et al.* Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology
Challenge 2019. *Critical Care Medicine* 2020;48(2):210–217. Data:
<https://physionet.org/content/challenge-2019/1.0.0/> (Creative Commons Attribution 4.0).
