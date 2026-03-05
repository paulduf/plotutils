# AUC Report

`AUCReport` is a linked, multi-variable AUC explorer that renders as a
self-contained HTML page.  It computes ROC curves and AUC values for
every (variable, outcome) combination and presents them in an interactive
2 x 2 grid:

| | Left | Right |
|---|------|-------|
| **Top** | AUC-AUC scatter (click a variable) | ROC curve for X-axis outcome |
| **Bottom** | Score distribution (boxplot or strip) | ROC curve for Y-axis outcome |

Clicking a point in the scatter plot updates the ROC curves and
distribution chart for the selected variable.

## Live demos

Explore the three interactive reports below.  Click any point in the
AUC-AUC scatter to update the linked panels.

- [Synthetic dataset](#synthetic-dataset) — 5 variables, 3 outcomes, 400 patients
- [Synthetic with missing data](#synthetic-with-missing-data) — same data with random null patterns
- [Diabetes dataset](#diabetes-dataset) — sklearn diabetes, 10 features, 3 progression thresholds

### Synthetic dataset

5 score variables with different affinities for 3 binary outcomes
(400 patients, no missing data):

<iframe
  src="../auc_report_synthetic.html"
  width="100%"
  height="750"
  style="border:none; overflow:hidden;"
  onload="var f=this;setTimeout(function(){f.style.height=f.contentDocument.documentElement.scrollHeight+'px'},1000)">
</iframe>

### Synthetic with missing data

Same synthetic data with randomly introduced null patterns (fixed seed).
Missing scores appear as cross marks below the distribution chart;
sample sizes and missing counts are shown in the scatter tooltip:

<iframe
  src="../auc_report_missing.html"
  width="100%"
  height="750"
  style="border:none; overflow:hidden;"
  onload="var f=this;setTimeout(function(){f.style.height=f.contentDocument.documentElement.scrollHeight+'px'},1000)">
</iframe>

### Diabetes dataset

sklearn diabetes dataset: 10 features (age, sex, bmi, bp, s1–s6) scored
against 3 binary outcomes defined by progression quartile thresholds
(Q25, Q50, Q75):

<iframe
  src="../auc_report_diabetes.html"
  width="100%"
  height="750"
  style="border:none; overflow:hidden;"
  onload="var f=this;setTimeout(function(){f.style.height=f.contentDocument.documentElement.scrollHeight+'px'},1000)">
</iframe>

## Basic usage

```python
from plotutils.auc import AUCReport

report = AUCReport(
    df,
    variables=["var_0", "var_1", "var_2"],
    outcomes=["outcome_0", "outcome_1"],
    id_col="patient_id",
)
report.to_html("report.html")
```

## Options

### Distribution style

Use `kind="strip"` for a jittered strip plot instead of the default
boxplot:

```python
report = AUCReport(df, variables, outcomes, kind="strip")
```

### Specificity levels

Annotate ROC curves at specific target specificity values.  Each level
draws a dashed cross-hair on the curve:

```python
report = AUCReport(df, variables, outcomes, specificity_levels=[0.80, 0.90, 0.95])
```

### Missing data handling

`AUCReport` handles missing data gracefully:

- **Missing outcome** — the row is excluded from AUC computation for
  that outcome.
- **Missing score** — the row is dropped from AUC computation and shown
  as a cross mark below the distribution chart.
- Sample sizes and missing counts are shown in the scatter plot tooltip.

## Reference

::: plotutils.auc.AUCReport
