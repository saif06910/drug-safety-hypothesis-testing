# Hypothesis Testing in Healthcare: Drug Safety

Statistical analysis of a randomized trial (Drug vs Placebo) focusing on adverse effects, counts, and age differences.

## What this project does
- Two-proportion z-test: compares **adverse effect** rates (Drug vs Placebo)
- Chi-square test: checks independence of **# of effects** and **treatment**
- Mann–Whitney U: compares **age** distributions (non-parametric)
- Outputs a short report at `results/summary.md` and a histogram at `docs/age_hist_by_group.png`

## How to run
```bash
pip install -r requirements.txt
python src/analysis.py
