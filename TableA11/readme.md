# Table A.11: Reproducible Statistical Analysis

Inputs, code, and results for all dataset-level comparisons in Table A.11 of the TAPFormer manuscript.

## Files

| File | Description |
| --- | --- |
| `inputs_table5.csv` | Paired dataset-level mean scores reported to one decimal place in Table 5. |
| `reproduce_A11.py` | Paired t-test and joint Holm-Bonferroni calculation functions. |
| `all_12_results.csv` | Mean paired differences, Cohen's d_z, t statistics, raw and adjusted p-values, and significance decisions. |
| `holm_ranked_steps.csv` | Jointly ranked raw p-values, Holm multipliers, products, and final adjusted p-values. |

## Reproduce

Requires Python, NumPy, SciPy, and statsmodels. Run from this directory:

```bash
python -m pip install numpy scipy statsmodels
python -c "from pathlib import Path; from reproduce_A11 import calculate; calculate(Path('inputs_table5.csv'), Path('.'))"
```

The command regenerates both output CSV files using only this release. Use it rather than `python reproduce_A11.py`, whose default entry point also runs supplementary analyses requiring additional inputs.

## Analysis

Two-sided paired t-tests use 10 datasets for image-level metrics and 11 for pixel-level metrics. Holm-Bonferroni correction is applied jointly. The calculations are checked against SciPy and statsmodels.

Mean differences are TAPFormer minus baseline, in percentage points. Calculations use the supplied one-decimal scores without intermediate rounding.
