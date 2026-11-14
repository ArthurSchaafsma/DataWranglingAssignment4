# Reweighing (Kamiran & Calders) on Adult, German and COMPAS

Assignment 4, INFOMDW (Utrecht University). Evaluation of the pre-processing algorithm **Reweighing**
(AI Fairness 360, `aif360.algorithms.preprocessing.Reweighing`) with DP_Diff, DIR, consistency, the extra measure
AEO_Diff (average equalized-odds difference) and accuracy / balanced accuracy / F1.

## Quick start (IntelliJ IDEA)
1. **Python plugin.** If IntelliJ offers no Python interpreter: *Settings > Plugins > Marketplace*, install the
   *Python* plugin (Community edition: *Python Community Edition*), restart.
2. **Open the project:** *File > Open* and choose this folder.
3. **Interpreter:** *File > Project Structure > SDKs > + > Add Python SDK > Virtualenv Environment > New*
   (location `.venv`, base interpreter Python 3.10-3.12; tested with 3.12), then set it as the *Project SDK*.
4. **Packages:** open the *Terminal* tool window (Alt+F12). It should show `(.venv)`. Run
   ```
   pip install -r requirements.txt
   ```
   (`pandas` must stay below 3: AIF360 0.6.1 crashes with pandas 3.x.)
5. **Run:** right-click `Reweighing_fairness_pipeline.ipynb` > *Run*. The first run downloads the three datasets into `data/raw/`
   (internet needed) and takes about 1-3 minutes in total. AIF360 prints a few warnings about optional packages
   (AdversarialDebiasing, fairlearn, inFairness): ignore them.

Without an IDE: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, `python Reweighing_fairness_pipeline.ipynb`.

## What you get (written next to the script)
| File | Content |
|---|---|
| `results_tables.tex` | the three LaTeX tables of the report (copy it into the Overleaf project, replacing the dummy file) |
| `results_reweighing.csv` | headline result (seed 42) |
| `results_reweighing_multiseed.csv`, `results_reweighing_summary.csv` | all runs / mean ± std over the 10 splits |
| `results_reweighing_variant_*.csv` | the same without the protected attribute as classifier input (paper's assumption A2) |
| `reweighing_weights.csv` | the four weights per dataset |

The last lines printed by the script give the **environment** (Python, scikit-learn, pandas, AIF360, OS) and the total
run time: copy them into the report's parameter section.

## Files
```
Reweighing_fairness_pipeline.ipynb     the pipeline as a notebook (needs `jupyter`; IntelliJ Ultimate or Jupyter)
loaders/                               AIF360 dataset loaders, copied unchanged (Apache-2.0)
requirements.txt
```
Settings are in the *Configuration* cell: test size, seeds,
classifier, `USE_SENSITIVE_AS_FEATURE` (main experiment: True), `RUN_A2_VARIANT`, `COMPAS_DROP_CHARGE_DESC`.

## Troubleshooting
* `ModuleNotFoundError: aif360` or `sklearn`: the packages were installed into another interpreter. Check that the
  Terminal shows `(.venv)` and that the project SDK is that `.venv`.
* A `TypeError` about string dtypes while loading a dataset: `pandas` is 3.x. Run `pip install "pandas<3"`.
* `pip install` fails on a very new Python (3.13+): create the virtualenv with Python 3.12.
* A dataset cannot be downloaded: the script prints the URL and the folder (`data/raw/<dataset>/`) to put the file in.

## Method (short)
Reweighing gives every training instance a weight `P(s) * P(y) / P(s, y)` (group `s`, label `y`), so that group and label
are independent in the weighted training data. A logistic regression is trained with these weights and compared with
the same model without weights on a held-out stratified test split, over 10 random splits.
