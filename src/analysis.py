# src/analysis.py
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import pingouin as pg

# ---- folders
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIGS = BASE / "docs"     # for images used in README
RESULTS = BASE / "results"
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ---- load
df = pd.read_csv(DATA / "drug_safety.csv")

# ---- A) Proportion of adverse effects (Drug vs Placebo)
# make a robust boolean "Yes"
is_yes = df["adverse_effects"].astype(str).str.lower().isin(["yes", "true", "1"])
counts_yes = is_yes.groupby(df["trx"]).sum().reindex(["Drug", "Placebo"])
group_n = df.groupby("trx").size().reindex(["Drug", "Placebo"])

stat, p_prop = proportions_ztest(count=counts_yes.values, nobs=group_n.values, alternative="two-sided")
prop_drug = counts_yes["Drug"] / group_n["Drug"]
prop_placebo = counts_yes["Placebo"] / group_n["Placebo"]

# ---- B) Independence: num_effects vs trx (chi-square)
chi_tables = pg.chi2_independence(data=df, x="num_effects", y="trx")
chi_stats = chi_tables[2].iloc[0]     # first row has global test
p_chi = chi_stats["pval"]

# ---- C) Age distribution by group
ax = sns.histplot(data=df, x="age", hue="trx", bins=30, stat="count", common_norm=False, kde=False)
ax.set_title("Age distribution by treatment group")
plt.tight_layout()
plt.savefig(FIGS / "age_hist_by_group.png", dpi=150)
plt.close()

# Normality check by group (Shapiro)
norm = pg.normality(data=df, dv="age", group="trx", method="shapiro", alpha=0.05)

# Since age is typically non-normal in trials, use Mann–Whitney U (safe choice)
age_drug = df.loc[df["trx"] == "Drug", "age"]
age_placebo = df.loc[df["trx"] == "Placebo", "age"]
mwu = pg.mwu(age_drug, age_placebo)
p_mwu = float(mwu["p-val"])

# ---- Save a short report (and print)
report = f"""
# Drug Safety – Hypothesis Tests

## Proportion of adverse effects
- Drug:    {prop_drug:.3f}  ({int(counts_yes['Drug'])}/{int(group_n['Drug'])})
- Placebo: {prop_placebo:.3f}  ({int(counts_yes['Placebo'])}/{int(group_n['Placebo'])})
- Two-sided z-test p-value: **{p_prop:.4g}**  (z = {stat:.3f})

## Independence (num_effects vs trx)
- Chi-square p-value: **{p_chi:.4g}**

## Age differences by group
- Mann–Whitney U p-value: **{p_mwu:.4g}**
- Histogram saved to: docs/age_hist_by_group.png
"""

print(report)
(RESULTS / "summary.md").write_text(report.strip(), encoding="utf-8")
