"""Reproduce all 12 dataset-level comparisons from explicit, published inputs.

Run: python reproduce_A11.py
Requires numpy, scipy, statsmodels. No unpublished logs or main.py are used.
Input scores are percentage scores. Each CSV row is one paired dataset observation.
"""
from pathlib import Path
import csv
import json
import platform

import numpy as np
import scipy
from scipy import stats
import statsmodels
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
BASELINES = ("AnomalyCLIP", "AdaCLIP", "FiLo")
METRICS = ("I-AUROC", "I-AP", "P-AUROC", "P-PRO")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calculate(input_file, output_dir):
    with input_file.open(encoding="utf-8-sig", newline="") as f:
        observations = list(csv.DictReader(f))
    results = []
    for baseline in BASELINES:
        for metric in METRICS:
            group = [r for r in observations
                     if r["baseline"] == baseline and r["metric"] == metric]
            n = len(group)
            assert n == (10 if metric.startswith("I-") else 11)
            assert len({r["dataset"] for r in group}) == n
            tap = np.array([float(r["tapformer"]) for r in group])
            base = np.array([float(r["baseline_score"]) for r in group])
            assert np.isfinite(tap).all() and np.isfinite(base).all()
            d = tap - base
            md = float(d.mean())
            sd = float(d.std(ddof=1))
            dz = md / sd
            t = md / (sd / np.sqrt(n))
            raw = float(2 * stats.t.sf(abs(t), df=n-1))
            independent = stats.ttest_rel(tap, base, alternative="two-sided")
            np.testing.assert_allclose([t, raw], [independent.statistic,
                                                 independent.pvalue], rtol=1e-12)
            results.append(dict(baseline=baseline, metric=metric, n=n, df=n-1,
                                mean_difference_pp=md, sd_difference_pp=sd,
                                cohens_dz=dz, t=t, raw_p=raw))
    assert len(observations) == 126 and len(results) == 12
    raw = np.array([r["raw_p"] for r in results])
    order = np.argsort(raw, kind="stable")
    m = len(results)  # One joint family: 3 baselines x 4 metrics.
    products = (m - np.arange(m)) * raw[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(products))
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    reference = multipletests(raw, alpha=0.05, method="holm")[1]
    np.testing.assert_allclose(adjusted, reference, rtol=1e-12, atol=1e-15)
    assert np.all(np.diff(adjusted_sorted) >= 0)
    assert np.all(adjusted >= raw) and np.all(adjusted <= 1)
    for i, row in enumerate(results):
        row["holm_adjusted_p"] = float(adjusted[i])
        row["significant_at_0_05"] = bool(adjusted[i] < 0.05)
    steps = []
    for j, i in enumerate(order):
        r = results[i]
        steps.append(dict(rank=j+1, baseline=r["baseline"], metric=r["metric"],
                          n=r["n"], df=r["df"], raw_p=r["raw_p"],
                          holm_multiplier=m-j, product=float(products[j]),
                          holm_adjusted_p=float(adjusted_sorted[j]),
                          significant_at_0_05=r["significant_at_0_05"]))
    output_dir.mkdir(exist_ok=True)
    write_csv(output_dir / "all_12_results.csv", results)
    write_csv(output_dir / "holm_ranked_steps.csv", steps)
    print(input_file.name, ":", sum(r["significant_at_0_05"] for r in results),
          "/ 12 significant; manual Holm agrees with statsmodels")
    return results, steps


def crosscheck_published_seed_scores():
    """Verify Appendix inputs and independently check the Table 6 p<0.0001 claim."""
    with (ROOT / "published_per_seed_scores.csv").open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    scores = {(r["seed"], r["dataset"], r["method"], r["metric"]): float(r["score"])
              for r in raw}
    assert len(raw) == len(scores) == 1680
    seeds = sorted({r["seed"] for r in raw}, key=int)
    assert len(seeds) == 10
    with (ROOT / "inputs_appendix.csv").open(encoding="utf-8-sig", newline="") as f:
        appendix_pairs = list(csv.DictReader(f))
    for row in appendix_pairs:
        for method, column in (("TAPFormer", "tapformer"),
                               (row["baseline"], "baseline_score")):
            reconstructed = np.mean([scores[seed, row["dataset"], method, row["metric"]]
                                     for seed in seeds])
            np.testing.assert_allclose(float(row[column]), reconstructed, atol=1e-12, rtol=0)
    run_results = []
    for baseline in BASELINES:
        for metric in METRICS:
            datasets = list(dict.fromkeys(r["dataset"] for r in raw if r["metric"] == metric))
            tap = [np.mean([scores[seed, d, "TAPFormer", metric] for d in datasets]) for seed in seeds]
            base = [np.mean([scores[seed, d, baseline, metric] for d in datasets]) for seed in seeds]
            test = stats.ttest_rel(tap, base, alternative="two-sided")
            run_results.append(dict(baseline=baseline, metric=metric, n_runs=10,
                                    raw_p=float(test.pvalue)))
    adjusted = multipletests([r["raw_p"] for r in run_results], method="holm")[1]
    for row, p in zip(run_results, adjusted):
        row["holm_adjusted_p"] = float(p)
    write_csv(ROOT / "run_level_crosscheck.csv", run_results)
    assert max(adjusted) < 0.0001
    print("Published per-seed cross-check: Appendix paired inputs agree;",
          "largest run-level Holm p =", float(max(adjusted)))


if __name__ == "__main__":
    for source in ("appendix", "table5"):
        calculate(ROOT / f"inputs_{source}.csv", ROOT / source)
    crosscheck_published_seed_scores()
    (ROOT / "software_versions.json").write_text(json.dumps(dict(
        python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, statsmodels=statsmodels.__version__), indent=2),
        encoding="utf-8")
