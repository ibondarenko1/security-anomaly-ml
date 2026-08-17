# Security Anomaly ML

An open-source ML-based network security anomaly detection system designed to convert high-volume network-flow detections into operational security incidents.

## Current status

The project contains a frozen temporal detection pipeline evaluated on a locked February 18 holdout. No training, threshold, aggregation, promotion, whitelisting, or suppression decision was changed after holdout access.

Current verdict: **Acceptable but operationally noisy.** The system is a research and engineering prototype, not a production-ready detector.

## Architecture

```mermaid
flowchart LR
    A[Network flows] --> B[Feature pipeline]
    B --> C[Context ML detector]
    C --> D[Attack score]
    D --> E[Threshold 0.10]
    E --> F[Incident aggregation]
    F --> G[Policy B / 5 min]
    G --> H[Incident promotion]
    H --> I[max score >= 0.25]
    I --> J[Analyst-facing incident]
```

## Key locked-holdout results

| Metric | Feb 18 result |
|---|---:|
| Flow recall | 98.3686% |
| Flow FPR | 2.1190% |
| Aggregated incident recall | 99.9917% |
| Promoted incident recall | 99.9339% |
| Promoted incident precision | 93.46% |
| End-to-end FP-object reduction vs raw flow alerts | 96.74% |

The flow-level detector slightly missed the pre-existing 98.5% recall research gate, while incident-level temporal aggregation generalized strongly. The remaining analyst-facing volume is still too high for a production Tier-1 queue.

## Quick start

Python 3.11 or newer is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the test suite:

```powershell
python -m pytest -q
```

The repository intentionally does not redistribute raw datasets, generated Parquet score dumps, or trained model binaries. Reproduce them locally from the documented scripts.

## Datasets

This project uses UNSW-NB15 and the CIC-UNSW-NB15 flow export for separate experiment stages. Dataset files are not included in this repository. Obtain them from their original publishers and comply with their original licenses and terms. The Apache-2.0 license in this repository applies to the original project source code, not to third-party datasets.

## Repository workflow

The detailed sections below preserve the experimental record from baseline modeling through temporal context, incident aggregation, promotion, and the locked holdout. Generated metrics and manifests may be reproduced locally; large binary artifacts are deliberately excluded from Git history.

## Preprocessing design

The development split is created only from the official UNSW-NB15 training set:

```text
Official training set (175,341 rows)
├── Development training: 80% (140,272 rows)
└── Validation: 20% (35,069 rows)

Official testing set (82,332 rows)
└── Final holdout; never included in development training or validation
```

The split is stratified by `label` and uses `random_state=42`. The model input excludes:

- `id`, because it is a technical row identifier;
- `attack_cat`, because it would leak target information;
- `label`, because it is the binary target.

The remaining 42 raw inputs contain 39 numeric and 3 categorical features. A scikit-learn `ColumnTransformer` applies:

- `StandardScaler` to numeric features;
- `OneHotEncoder(handle_unknown="ignore")` to `proto`, `service`, and `state`.

`service="-"` is retained as a legitimate category. The preprocessor is fitted only on the development training subset. Validation and official test data are transformed but never used to fit preprocessing. The transformed matrices remain in memory and are not saved as CSV files.

Run the preprocessing checks and report with:

```powershell
python src/preprocess.py
```

No duplicate-looking flows are removed during preprocessing. No feature selection, PCA, resampling, or class balancing is performed.

## Baseline training and validation

The first supervised baseline stage uses only the development training and validation subsets. The official 82,332-row UNSW testing set remains unused and is reserved for final model selection.

Exactly two classifiers are evaluated at the default classification threshold of `0.5`:

1. `DummyClassifier(strategy="most_frequent")` as a non-ML reference.
2. `LogisticRegression(max_iter=1000, solver="lbfgs")` as a simple untuned linear baseline.

Neither model uses class weights, resampling, feature selection, PCA, threshold optimization, or hyperparameter tuning. Validation results are:

| Model | Accuracy | Attack precision | Attack recall | Attack F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0.680630 | 0.680630 | 1.000000 | 0.809970 | 0.500000 | 0.680630 | 1.000000 | 0.000000 |
| LogisticRegression | 0.936440 | 0.923682 | 0.988269 | 0.954885 | 0.984301 | 0.991744 | 0.174018 | 0.011731 |

Logistic Regression validation confusion matrix:

```text
True Normal predicted Normal:  9,251
True Normal predicted Attack:  1,949
True Attack predicted Normal:    280
True Attack predicted Attack: 23,589
```

The logistic baseline meaningfully outperforms the Dummy reference, particularly in ROC-AUC, PR-AUC, F1, and false-positive rate. However, a 17.40% validation false-positive rate and 280 missed attacks mean it is not production-ready.

Run the reproducible baseline experiment with:

```powershell
python src/train_baseline.py
```

The stage writes:

- `models/baseline_metrics.json`;
- `models/baseline_logistic_regression.joblib`;
- `models/preprocessor.joblib`.

## Random Forest comparison

The next comparison stage trains exactly one nonlinear candidate:

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)
```

It reuses the saved training-fitted `models/preprocessor.joblib` without refitting. No tuning, class weighting, resampling, feature selection, PCA, or threshold optimization is performed. The official UNSW test set remains unused.

Training metrics, used only to inspect overfitting:

| Accuracy | Attack precision | Attack recall | Attack F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.998339 | 0.998284 | 0.999277 | 0.998780 | 0.999977 |

Direct validation comparison at the default `0.5` threshold:

| Metric | Logistic Regression | Random Forest | RF − Logistic |
|---|---:|---:|---:|
| Precision | 0.923682 | 0.963671 | +0.039989 |
| Recall | 0.988269 | 0.977963 | -0.010306 |
| F1 | 0.954885 | 0.970764 | +0.015879 |
| ROC-AUC | 0.984301 | 0.993948 | +0.009647 |
| PR-AUC | 0.991744 | 0.997109 | +0.005366 |
| FPR | 0.174018 | 0.078571 | -0.095446 |
| FNR | 0.011731 | 0.022037 | +0.010306 |
| False positives | 1,949 | 880 | -1,069 |
| False negatives | 280 | 526 | +246 |

Random Forest validation confusion matrix:

```text
True Normal predicted Normal: 10,320
True Normal predicted Attack:    880
True Attack predicted Normal:    526
True Attack predicted Attack: 23,343
```

The forest cuts false alerts by 1,069 and reduces FPR from 17.40% to 7.86%, while recall remains high at 97.80%. The tradeoff is 246 additional missed attacks. It is the current leading validation candidate on overall operational balance, but the training-validation gap shows overfitting and the model is not production-ready.

Top Random Forest importance signals include `ct_state_ttl`, `sttl`, `dload`, `rate`, `dttl`, `sload`, `dur`, `tcprtt`, and packet-timing/volume features. These importances are diagnostic only; no feature selection has been performed.

Run the comparison with:

```powershell
python src/train_random_forest.py
```

The stage writes `models/random_forest.joblib` and `models/random_forest_metrics.json` without overwriting the Logistic Regression artifacts.

## Random Forest hyperparameter tuning

Random Forest structure is tuned only within the 140,272-row development training subset. The search uses 12 reproducibly sampled configurations and:

```python
StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
RandomForestClassifier(n_estimators=200, random_state=42)
```

The search evaluates attack precision, recall, F1, ROC-AUC, PR-AUC, and specificity. It first requires mean CV attack recall of at least `0.975`, then minimizes mean CV FPR; effectively tied FPR results are ordered by PR-AUC and F1. Accuracy is not used for candidate selection. The 35,069-row validation subset is evaluated exactly once after selection, and the official test set remains completely unused.

Selected structural hyperparameters:

```text
max_depth=24
min_samples_split=5
min_samples_leaf=2
max_features=0.5
max_samples=0.8
bootstrap=True
```

The selected candidate reached mean CV recall `0.977857`, mean CV FPR `0.087366`, PR-AUC `0.996784`, and F1 `0.968726`. It was retrained on the complete development training subset with `n_estimators=300` and the default classification threshold of `0.5`.

Final validation comparison:

| Metric | Logistic Regression | Untuned RF | Tuned RF |
|---|---:|---:|---:|
| Precision | 0.923682 | 0.963671 | 0.964539 |
| Recall | 0.988269 | 0.977963 | 0.978885 |
| F1 | 0.954885 | 0.970764 | 0.971659 |
| ROC-AUC | 0.984301 | 0.993948 | 0.994026 |
| PR-AUC | 0.991744 | 0.997109 | 0.997160 |
| FPR | 0.174018 | 0.078571 | 0.076696 |
| FNR | 0.011731 | 0.022037 | 0.021115 |
| False positives | 1,949 | 880 | 859 |
| False negatives | 280 | 526 | 504 |

Tuned Random Forest validation confusion matrix:

```text
True Normal predicted Normal: 10,341
True Normal predicted Attack:    859
True Attack predicted Normal:    504
True Attack predicted Attack: 23,365
```

Tuning reduced the train-to-validation gaps: accuracy from `0.038431` to `0.029919`, F1 from `0.028016` to `0.021790`, and recall from `0.021314` to `0.017847`. Relative to untuned RF, it eliminated 21 false alerts and recovered 22 attacks while slightly improving ranking metrics. It is therefore the current leading validation candidate, but it is not production-ready.

The randomized search took `1465.317478` seconds, final training took `159.783054` seconds, and validation inference took `0.460869` seconds. No search or final-training warnings were recorded.

Run this stage with:

```powershell
python src/tune_random_forest.py
```

The stage writes without overwriting previous model artifacts:

- `models/tuned_random_forest.joblib`;
- `models/tuned_random_forest_metrics.json`;
- `models/random_forest_tuning_results.csv`.

No threshold optimization, class balancing, resampling, PCA, or feature selection is performed. The official UNSW test set is not loaded, transformed, or evaluated.

## Operational threshold selection

The tuned Random Forest architecture and structural hyperparameters are frozen. Its operational attack-score threshold is selected from out-of-fold (OOF) scores over all 175,341 rows of the official UNSW training set, rather than from the previously used 35,069-row validation subset.

The procedure uses:

```python
StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
```

For each fold, a new `StandardScaler` and `OneHotEncoder(handle_unknown="ignore")` are fitted only on that fold's training rows. A frozen tuned Random Forest is then fitted on those transformed training rows and scores only the held-out fold. Every official training row receives exactly one score from a model that did not train on that row. Fold models and preprocessors are temporary and are not saved.

The `predict_proba()` class-1 output is treated as an **attack score**, not as a calibrated real-world probability. Probability calibration has not been performed.

Thresholds from `0.10` through `0.90` in steps of `0.01` are evaluated using the rule `attack_score >= threshold`. The operational policy first requires attack recall of at least `0.985`; among qualifying thresholds it minimizes FPR, with precision, F1, and the higher threshold as tie-breakers. Accuracy is not used for selection.

All 175,341 rows were scored exactly once. OOF score-level results were ROC-AUC `0.993788` and PR-AUC / Average Precision `0.997013`. The selected threshold is `0.45`:

| Threshold | Attack precision | Attack recall | F1 | FPR | FNR | False positives | False negatives |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 0.954872 | 0.985261 | 0.969828 | 0.099232 | 0.014739 | 5,557 | 1,759 |
| 0.50 | 0.961165 | 0.979705 | 0.970346 | 0.084357 | 0.020295 | 4,724 | 2,422 |

Relative to threshold `0.50`, threshold `0.45` recovers 663 attacks and satisfies the 98.5% recall requirement, at the cost of 833 additional false alerts. FPR increases by `0.014875`, while FNR decreases by `0.005556`. This is the explicit operational trade-off selected by the stated security policy.

The complete three-fold run took `450.237221` seconds and produced no warnings. The official UNSW test set was not loaded, transformed, or evaluated, and no final all-training-data model was trained.

Run this stage with:

```powershell
python src/select_threshold.py
```

The stage writes:

- `models/threshold_results.csv`, containing all 81 threshold candidates;
- `models/selected_threshold.json`, containing the selected policy, OOF metrics, frozen hyperparameters, CV audit, and reproducibility metadata.

## Final locked evaluation

Model selection and OOF threshold selection were completed before official test access. The frozen preprocessor and Random Forest were fitted on all 175,341 official training flows and saved before the 82,332-row official test set was loaded. The test set was then used once for the locked final evaluation. No model, feature, preprocessing, sampling, hyperparameter, or threshold tuning was performed after test results became available.

The final training-fitted preprocessor maps 42 raw input features to 194 encoded features. Final training took `257.283021` seconds, including `0.582571` seconds for preprocessing and `256.686997` seconds for Random Forest fitting. No training warnings were recorded.

Official test results at the locked operational threshold `0.45`:

| Metric | Official test |
|---|---:|
| Accuracy | 0.859532 |
| Attack precision | 0.800820 |
| Attack recall | 0.991485 |
| Attack F1 | 0.886011 |
| ROC-AUC | 0.981233 |
| PR-AUC / Average Precision | 0.985756 |
| Specificity | 0.697865 |
| FPR | 0.302135 |
| FNR | 0.008515 |

```text
True Normal predicted Normal: 25,821
True Normal predicted Attack: 11,179
True Attack predicted Normal:    386
True Attack predicted Attack: 44,946
```

The model detected 44,946 of 45,332 attacks (`99.1485%`) and missed 386. It incorrectly alerted on 11,179 of 37,000 normal flows (`30.2135%`). The attack-recall target of at least `98.5%` therefore survived, but false-alert control did not generalize to the official test distribution.

Comparison with the OOF threshold-selection estimate:

| Metric | OOF estimate | Official test | Test − OOF |
|---|---:|---:|---:|
| Precision | 0.954872 | 0.800820 | -0.154053 |
| Recall | 0.985261 | 0.991485 | +0.006224 |
| F1 | 0.969828 | 0.886011 | -0.083818 |
| ROC-AUC | 0.993788 | 0.981233 | -0.012555 |
| PR-AUC | 0.997013 | 0.985756 | -0.011257 |
| FPR | 0.099232 | 0.302135 | +0.202903 |
| FNR | 0.014739 | 0.008515 | -0.006224 |

The test FPR increased by 20.29 percentage points and precision fell by 15.41 points. Under the predeclared consistency rule—no absolute OOF-to-test difference above `0.02` for the seven comparison metrics—the final test is not consistent with OOF expectations.

Threshold `0.50` was calculated only as a post-evaluation diagnostic and did not alter the locked `0.45` decision:

| Threshold | Precision | Recall | FPR | FNR | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.45 — locked | 0.800820 | 0.991485 | 0.302135 | 0.008515 | 11,179 | 386 |
| 0.50 — diagnostic | 0.816453 | 0.986941 | 0.271838 | 0.013059 | 10,058 | 592 |

Per-category binary detection diagnostics at threshold `0.45`:

| Attack category | Flows | Detected | Missed | Detection recall |
|---|---:|---:|---:|---:|
| Generic | 18,871 | 18,871 | 0 | 1.000000 |
| Exploits | 11,132 | 11,103 | 29 | 0.997395 |
| Fuzzers | 6,062 | 5,713 | 349 | 0.942428 |
| DoS | 4,089 | 4,087 | 2 | 0.999511 |
| Reconnaissance | 3,496 | 3,495 | 1 | 0.999714 |
| Analysis | 677 | 675 | 2 | 0.997046 |
| Backdoor | 583 | 583 | 0 | 1.000000 |
| Shellcode | 378 | 376 | 2 | 0.994709 |
| Worms | 44 | 43 | 1 | 0.977273 |

The aggregate recall is strong, but `Fuzzers` recall (`94.24%`) is materially weaker than the other attack categories. This diagnostic did not trigger retraining or tuning.

Total official-test transform and model-scoring time was `0.879352` seconds. Reloading the final model and preprocessor reproduced all locked predictions exactly; attack scores matched within machine precision with a maximum absolute difference of `4.44e-16`.

Run the locked training and evaluation with:

```powershell
python src/train_final.py
```

The final evaluation writes:

- `models/final_random_forest.joblib`;
- `models/final_preprocessor.joblib`;
- `models/final_test_metrics.json`;
- `models/final_attack_category_metrics.csv`;
- `models/final_model_manifest.json`.

The model generalizes well for attack ranking and high recall, but it does not generalize adequately for operational false-positive control. It is therefore a completed experimental final candidate, not a deployment-ready detector.

## Post-hoc distribution-shift investigation

The official test result is investigated diagnostically without changing the model, features, preprocessing, hyperparameters, threshold, or calibration. Individual OOF scores are regenerated using the same three fold-local preprocessing/model fits and matched back to all 175,341 official training rows. The already-locked final model supplies official-test scores. The regenerated classifications reproduce the locked OOF and test counts exactly; only ranking metrics show machine-scale tie variation below `1e-8`.

The class-conditional score analysis confirms that the FPR increase is primarily a **normal-score distribution shift**, not a broad shift of both classes:

| Score group | Mean | Median | 75th percentile | Share ≥ 0.45 |
|---|---:|---:|---:|---:|
| OOF normal | 0.093912 | 0.000000 | 0.000000 | 0.099232 |
| Official-test normal | 0.250196 | 0.000000 | 0.535947 | 0.302135 |
| OOF attack | 0.954376 | 1.000000 | 1.000000 | 0.985261 |
| Official-test attack | 0.958864 | 0.999981 | 1.000000 | 0.991485 |

Normal-score OOF-to-test drift is large: KS statistic `0.263897`, PSI `0.333764`, and mean score shift `+0.156283`. Attack-score drift is much smaller: KS `0.124066`, PSI `0.008168`, and mean shift `+0.004488`. This explains why recall transfers while FPR does not.

Score reliability also degrades substantially:

| Diagnostic | OOF | Official test |
|---|---:|---:|
| Brier score | 0.028749 | 0.082598 |
| 10-bin expected calibration error | 0.006804 | 0.089788 |

No calibration model was fitted. These values only diagnose that the absolute Random Forest score scale transfers poorly to the official test distribution.

The largest raw-feature shifts among normal flows are concentrated in `sttl`, `ct_state_ttl`, `ackdat`, `dload`, `dttl`, `tcprtt`, `dmean`, and `synack`. For example, false-positive median `sttl` is `254`, compared with `31` for both OOF true negatives and official-test true negatives. Median `dttl` similarly changes from `29` to `252`, while median `tcprtt` changes from approximately `0.0007` to `0.1296`.

False positives are highly clustered rather than uniformly distributed:

| proto | service | state | Normal flows | False positives | Cluster FPR | Share of all FP |
|---|---|---|---:|---:|---:|---:|
| tcp | `-` | FIN | 17,644 | 6,364 | 0.360689 | 0.569282 |
| udp | `-` | INT | 3,497 | 2,981 | 0.852445 | 0.266661 |
| tcp | http | FIN | 4,001 | 1,453 | 0.363159 | 0.129976 |
| tcp | ftp | FIN | 753 | 353 | 0.468792 | 0.031577 |

The first three clusters account for `96.6%` of all 11,179 false positives; the first four account for `99.7%`. `service="-"` remains a legitimate category, but it appears in 83.8% of false alerts and therefore identifies an important traffic segment for investigation rather than a missing-value issue.

The `Fuzzers` weakness is also localized. Its 349 misses represent `90.4%` of all official-test false negatives. The strongest raw-feature differences between detected and missed Fuzzers include `ct_dst_src_ltm`, `smean`, `ct_srv_dst`, `ct_src_dport_ltm`, and `ct_dst_ltm`. Of the missed Fuzzers, 340 (`97.4%`) have `service="-"`; 251 use TCP and 98 use UDP.

Run the read-only diagnostic stage with:

```powershell
python src/investigate_shift.py
```

Key outputs are stored under `models/diagnostics/`, including per-row OOF/test scores, numeric and categorical drift tables, FP clusters, Fuzzers diagnostics, score-distribution plots, and reliability plots. The machine-readable investigation summary is `models/shift_investigation.json`.

Because official-test labels have now informed this investigation, they cannot be used to select and honestly re-evaluate a calibration method, threshold, or model change. Any remediation designed from these findings requires a new untouched holdout for an unbiased final assessment.

### Locked v1 generalization package

The Random Forest is not incrementally trained or modified after this diagnosis. The saved final model, preprocessor, and threshold `0.45` are fixed as the **v1 baseline**. Run the packaged diagnosis with:

```powershell
python src/diagnose_generalization.py
```

This produces:

- `models/final_score_distribution_summary.csv`, with mean, median, p75, p90, p95, p99, and crossing rates for thresholds `0.45` and `0.50` for OOF/test normal and attack scores;
- `models/final_feature_drift.csv`, one drift-ranked row for every 39 numeric and 3 categorical input features;
- `models/final_categorical_frequency_drift.csv`, including category frequency changes and unseen test categories;
- `models/final_false_positives.csv`, all 11,179 false-positive flows with original columns, attack score, and predicted class;
- `models/final_fp_vs_true_normal.csv` and `models/final_fp_traffic_clusters.csv`, comparing false positives with correctly classified normal traffic;
- `models/final_fuzzers_comparison.csv` and `models/final_fuzzers_predictions.csv`, comparing 349 missed Fuzzers with 5,713 detected Fuzzers;
- `models/generalization_diagnosis.json`, the machine-readable v1 diagnosis and artifact hashes.

The official test contains two previously unseen `state` levels, `ACC` and `CLO`, but only five flows use them, so they do not explain the 11,179 false positives. The dominant evidence remains strong normal-class numeric drift and concentration in a small number of `proto/service/state` profiles.

The next model must be trained from scratch as **v2**, not fitted on top of v1. Its development acceptance gate is fixed before implementation:

```text
Validation attack recall >= 98.5%
Validation false-positive rate < 10%
Final evaluation on a new untouched locked test
```

The earlier idea of another row-only UNSW candidate is superseded by the
separate temporal-context v2 experiment below. The v1 evidence and acceptance
gate remain historical baseline criteria, not permission to alter v1.

## Security Anomaly ML v2: Temporal Context Network Triage

V1 remains frozen and unchanged. V2 is a separate production-oriented
experiment based on the CIC-UNSW-NB15 flow export produced by the Canadian
Institute for Cybersecurity from the original UNSW-NB15 packet captures. No v1
model, preprocessor, threshold, metric, or official-test artifact is reused for
v2 model selection.

### Dataset acquisition and inspection

The official dataset description is:

```text
https://www.unb.ca/cic/datasets/cic-unsw-nb15.html
```

The CIC download endpoint requires a registration form. The files currently in
`data/raw/cic_unsw_nb15/` were acquired from a public mirror of that package;
the mirror URL, filenames, byte sizes, and local SHA-256 hashes are recorded in
`models/cic_unsw_nb15_inspection.json`. The raw CSV files were not modified.

The package contains two materially different representations:

| File | Rows | Columns | Temporal/endpoint metadata | Intended v2 use |
|---|---:|---:|---|---|
| `CICFlowMeter_out.csv` | 3,540,241 | 84 | Yes | Full v2 source |
| `Data.csv` | 447,915 | 76 | No | Not suitable for temporal context |

The full export has seven metadata columns (`Flow ID`, `Src IP`, `Src Port`,
`Dst IP`, `Dst Port`, `Protocol`, and `Timestamp`), 76 numeric CICFlowMeter
features, and `Label`. It contains 3,450,658 benign flows (97.4696%) and 89,583
attack flows (2.5304%). All nine UNSW attack categories are present. Protocol
values in this export are limited to TCP (`6`) and UDP (`17`). No missing cells
or infinite numeric values were found during the complete chunked scan.

The published timestamp strings parse with the format
`%d/%m/%Y %I:%M:%S %p`. Their observed precision is one second; fractional
seconds and timezone information are absent. The range is
`2015-01-22 07:49:41` through `2015-02-18 08:29:24`, with 87,240 distinct
timestamp seconds. As many as 776 flows share one timestamp second, and
3,540,194 of the 3,540,241 rows share their timestamp with at least one other
flow. The CSV is not chronologically ordered.

Only three capture dates are represented:

| Capture date | Flows | Benign | Attacks |
|---|---:|---:|---:|
| 2015-01-22 | 1,765,922 | 1,751,442 | 14,480 |
| 2015-02-17 | 498,890 | 478,524 | 20,366 |
| 2015-02-18 | 1,275,429 | 1,220,692 | 54,737 |

This creates a natural chronological split candidate, but it is not yet a
locked split. Endpoint hosts overlap heavily across capture dates: pairwise
overlap is 35 to 39 IPs, while the complete dataset contains only 40 source IPs
and 39 destination IPs. A simultaneously strict host-disjoint and chronological
three-way split would therefore discard or isolate most of the experiment.
Before creating the split, the project must explicitly choose how to handle this
constraint. Raw IP addresses must not be supplied to the model as memorization
features.

Second-resolution timestamps also require a tie-safe causal policy: flows with
the same timestamp cannot be internally ordered reliably. Context features for
time `t` must use only state from timestamps strictly earlier than `t`; the
flows at `t` must be updated as one batch after all of them are scored. Context
state must be reset at split boundaries.

The CSV supports connection counts, distinct destinations and ports, repeated
source-to-destination activity, connection rates, rolling packet/byte features,
directionality, inter-arrival behavior, flag behavior, and burstiness. It does
not contain packet TTL, true RTT, the original UNSW `service` value, or the
original UNSW `state` value. `Flow IAT` must not be mislabeled as RTT. TTL/RTT
host deviations would require returning to the packet captures or another
packet-derived export; service can only be inferred cautiously from ports.

Run the read-only, memory-bounded inspection with:

```powershell
python src/explore_cic_unsw_nb15.py
```

This stage does not create a split, engineer features, preprocess inputs, or
train a model.

### Frozen v2 split and temporal feature build

The v2 capture-day split is now fixed:

```text
2015-01-22 -> train          (1,765,922 flows)
2015-02-17 -> validation     (  498,890 flows)
2015-02-18 -> locked holdout (1,275,429 flows)
```

Host overlap is allowed because later monitoring periods realistically observe
many of the same hosts. Identity memorization is not allowed: `Flow ID`, `Src
IP`, `Dst IP`, and `Timestamp` are used only while constructing history and are
absent from every model-ready output.

`src/build_temporal_features.py` performs a disk-backed external sort using
only `Timestamp`. Each capture day is evaluated in a separate query and starts
with empty context. Window frames end at `CURRENT ROW EXCLUDE GROUP`, so every
flow at timestamp `t` sees only rows with timestamps strictly earlier than `t`.
No state from train enters validation, and no train or validation state enters
the locked holdout.

The build produces three compressed Parquet datasets:

| Split | Rows | Model features | Targets | File size |
|---|---:|---:|---:|---:|
| Train | 1,765,922 | 128 | 2 | 236,488,850 bytes |
| Validation | 498,890 | 128 | 2 | 68,790,803 bytes |
| Locked holdout | 1,275,429 | 128 | 2 | 175,079,548 bytes |

The feature contract separates the two future experiments:

- **Baseline v2:** the original 76 numeric CICFlowMeter features only.
- **Context v2:** the same 76 features, nine numeric port/protocol behavioral
  representations, and 43 temporal context features.

The temporal features include 10/60-second source connection counts, distinct
destinations and destination ports, repeated source/destination and
source/port counts, rolling bytes and packets, 60-second means, time since the
last source and source/destination flow, protocol ratios, SYN/RST activity, and
port diversity. Destination-side analogues cover connection counts, distinct
sources/source ports, rolling volume, time since last destination flow,
protocol ratios, flags, and port diversity.

Rolling byte and packet volume use total bidirectional flow volume:

```text
flow_bytes   = Total Length of Fwd Packet + Total Length of Bwd Packet
flow_packets = Total Fwd Packet + Total Bwd packets
```

A seconds-since value of `-1` explicitly represents cold start/no prior event.
Ports are retained numerically and accompanied by one-hot-like flags for
well-known (`0-1023`), registered (`1024-49151`), and ephemeral
(`49152-65535`) ranges.

All generated datasets were checked for exact row counts, binary targets,
schema consistency, forbidden identifier columns, and missing/infinite context
values. The complete contract, file hashes, label mappings, and build settings
are stored in `data/processed/cic_unsw_nb15_v2/feature_manifest.json`.

Run the reproducible build with:

```powershell
python src/build_temporal_features.py --memory-limit 10GB --threads 8
```

The build does not fit preprocessing, train a model, select a threshold, or
evaluate the locked holdout. The next experiment must first compare the
76-feature baseline and 128-feature context candidate on the same February 17
validation set. Candidate selection remains: require attack recall of at least
98.5%, then minimize FPR, while also reporting precision, PR-AUC, and false
positives per 100,000 benign flows.

### V2 temporal-context ablation

The 76-feature baseline and 128-feature context candidate were trained on the
same January 22 rows with identical Random Forest settings. No class weighting,
resampling, calibration, threshold tuning outside validation, or hyperparameter
tuning was used. Both models use:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=24,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.5,
    max_samples=0.8,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
```

Each model produced a complete attack-score vector for all 498,890 February 17
validation flows. Scores are uncalibrated Random Forest attack scores, not
real-world probabilities. Thresholds `0.01` through `0.99` were evaluated in
steps of `0.01`. The selected operating point is the threshold with the fewest
false positives among thresholds achieving attack recall of at least 98.5%.

| Validation metric | Baseline v2 | Context v2 | Context difference |
|---|---:|---:|---:|
| Selected threshold | 0.06 | 0.10 | +0.04 |
| Recall | 0.985957 | 0.985908 | -0.000049 |
| Precision | 0.508612 | 0.624269 | +0.115657 |
| F1 | 0.671056 | 0.764477 | +0.093422 |
| ROC-AUC | 0.989801 | 0.994684 | +0.004883 |
| PR-AUC / average precision | 0.781595 | 0.847746 | +0.066151 |
| Specificity | 0.959459 | 0.974745 | +0.015287 |
| FPR | 0.040541 | 0.025255 | -0.015287 |
| FNR | 0.014043 | 0.014092 | +0.000049 |
| False positives | 19,400 | 12,085 | -7,315 |
| False negatives | 286 | 287 | +1 |
| FP per 100k benign flows | 4,054.13 | 2,525.47 | -1,528.66 |

Context therefore reduces false positives by 37.7% relative while preserving
the operational recall gate, and it materially improves both ranking and the
selected operating point. This supports the temporal-context hypothesis for the
February 17 validation period. It is not yet production-ready: 2,525 false
alerts per 100,000 benign flows remains a high Tier-1 workload.

The aggregate result hides an important category tradeoff. Missed Fuzzers
increase from 182 to 261, reducing Fuzzers recall from 97.16% to 95.93%.
Analysis misses increase from 4 to 19. These regressions are offset by fewer
misses for Exploits, Reconnaissance, Shellcode, DoS, Backdoor, and Generic, so
the aggregate false-negative count changes by only one. Future work must use
January 22 and February 17 only to investigate this weakness; it must not adapt
features or thresholds using February 18.

Training took 1,358.7 seconds for baseline v2 and 1,119.3 seconds for context
v2. Validation inference took 2.32 and 2.53 seconds respectively. Neither model
emitted a warning, and reloaded artifacts reproduced checked validation scores
within machine precision.

Run the ablation with:

```powershell
python src/train_v2_ablation.py
```

The generated artifacts are:

- `models/v2_baseline_random_forest.joblib`;
- `models/v2_context_random_forest.joblib`;
- `models/v2_ablation_metrics.json`;
- `models/v2_threshold_results.csv`;
- `models/v2_ablation_comparison.csv`;
- `models/v2_attack_category_metrics.csv`;
- `models/v2_validation_scores.parquet`.

The February 18 locked-holdout Parquet is not referenced or loaded by the
ablation module. It remains unevaluated.

### V2 error overlap and score ensemble

The next experiment reused the saved February 17 validation scores. It did not
retrain either Random Forest. Error overlap is defined at each model's selected
operating point: threshold `0.06` for baseline and `0.10` for context.

Attack decisions overlap as follows:

| Outcome | Flows |
|---|---:|
| Both models detect the attack | 19,821 |
| Baseline detects, Context misses | 259 |
| Baseline misses, Context detects | 258 |
| Both models miss | 28 |

The errors are therefore highly complementary at the decision level. For
Fuzzers specifically, Context misses 261 flows; baseline catches 234 of them
(89.66%). Baseline also misses 182 Fuzzers, of which Context recovers 155. Only
27 Fuzzers are missed by both selected operating points.

Benign decisions show why Context is still the stronger standalone filter:

| Outcome | Flows |
|---|---:|
| Both true negative | 457,921 |
| False positive from both | 10,882 |
| False positive from baseline only | 8,518 |
| False positive from Context only | 1,203 |

A score ensemble was then evaluated without new training. For
`score = alpha * context_score + (1 - alpha) * baseline_score`, `alpha` was
swept from `0.00` through `1.00` in steps of `0.05`. The diagnostic
`max(baseline_score, context_score)` was also evaluated. Each candidate used a
fresh threshold sweep from `0.01` through `0.99`; selection retained the same
policy: achieve recall of at least 98.5%, then minimize FPR.

| Validation metric | Context | Selected blend | Difference |
|---|---:|---:|---:|
| Context weight (`alpha`) | 1.00 | 0.95 | -0.05 |
| Threshold | 0.10 | 0.12 | +0.02 |
| Recall | 0.985908 | 0.985466 | -0.000442 |
| Precision | 0.624269 | 0.626796 | +0.002526 |
| F1 | 0.764477 | 0.766235 | +0.001758 |
| ROC-AUC | 0.994684 | 0.994833 | +0.000150 |
| PR-AUC | 0.847746 | 0.855833 | +0.008087 |
| FPR | 0.025255 | 0.024973 | -0.000282 |
| False positives | 12,085 | 11,950 | -135 |
| False negatives | 287 | 296 | +9 |
| FP per 100k benign | 2,525.47 | 2,497.26 | -28.21 |

The selected blend passes the recall gate but does not reach the research target
of FPR below 2%. Its gain over Context is marginal: 135 fewer false positives
at the cost of nine additional missed attacks. A potentially useful Pareto
alternative is `alpha=0.85`, threshold `0.15`: it has recall `0.987234`, FPR
`0.025175`, 12,047 false positives, and 260 false negatives. It improves both
FP and FN counts relative to standalone Context, but the fixed selection policy
correctly selects `alpha=0.95` because its FPR is lower. The maximum-score
diagnostic is worse operationally: recall `0.985073`, FPR `0.031668`, 15,154
false positives, and 304 false negatives.

The 234 Fuzzers caught by baseline but missed by Context were compared with the
5,997 Fuzzers caught by both models using all 52 added features. The largest
distribution shifts occur in `dst_unique_sport_60s`, `src_dst_conn_60s`,
`src_dport_conn_60s`, `dst_conn_60s`, rolling packets per flow, and the source
and destination TCP/UDP ratios. These regression flows tend to occur in denser,
more repetitive, predominantly UDP context with fewer packets per flow. This is
an association and score-delta diagnostic, not a causal or SHAP attribution.

Run the analysis in the project environment with:

```powershell
.\.venv\Scripts\python.exe src\analyze_v2_ensemble.py
```

The generated artifacts are:

- `models/v2_error_overlap.json`;
- `models/v2_error_overlap_rows.parquet`;
- `models/v2_ensemble_threshold_results.csv`;
- `models/v2_ensemble_comparison.csv`;
- `models/v2_ensemble_attack_category_metrics.csv`;
- `models/v2_ensemble_validation_scores.parquet`;
- `models/v2_selected_ensemble.json`;
- `models/v2_fuzzer_context_regressions.parquet`;
- `models/v2_fuzzer_context_feature_diagnostics.csv`.

The February 18 locked holdout is absent from the analysis data paths and
remains untouched. The experiment shows that the models contain complementary
signals, but simple score averaging does not yet exploit enough of that signal
to justify opening the holdout.

### V2 Fuzzer-only sample weighting

The Context v2 architecture, 128-feature contract, Random Forest
hyperparameters, January 22 training rows, and February 17 validation rows were
kept fixed. The only training change was `sample_weight` for the 5,627 January
22 Fuzzer flows. Weights `1.00`, `1.25`, `1.50`, `2.00`, and `3.00` were tested;
all benign flows and every other attack category retained weight `1.00`.
`attack_cat` was used only to construct the training weight vector and was not
included in model input.

Each independently trained model produced a complete February 17 score vector.
For each vector, thresholds `0.01` through `0.99` were swept under the unchanged
policy: first require overall attack recall of at least 98.5%, then minimize
FPR. No stacking, calibration, feature changes, or Analysis weighting was used.

| Candidate | Threshold | Recall | FPR | FP | Overall FN | Fuzzer FN | Fuzzer recall | Analysis FN | PR-AUC | FP/100k benign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Context reference | 0.10 | 0.985908 | 0.025255 | 12,085 | 287 | 261 | 0.959301 | 19 | 0.847746 | 2,525.47 |
| Weight 1.00 control | 0.09 | 0.986448 | 0.025637 | 12,268 | 276 | 253 | 0.960549 | 19 | 0.848812 | 2,563.72 |
| Weight 1.25 | 0.10 | 0.985171 | 0.025428 | 12,168 | 302 | 271 | 0.957742 | 22 | 0.843368 | 2,542.82 |
| Weight 1.50 | 0.12 | 0.985024 | **0.024686** | **11,813** | 305 | 270 | 0.957898 | 22 | 0.837702 | **2,468.63** |
| Weight 2.00 | 0.11 | 0.986448 | 0.025464 | 12,185 | **276** | **244** | **0.961952** | 23 | 0.839534 | 2,546.37 |
| Weight 3.00 | 0.12 | 0.985859 | 0.025309 | 12,111 | 288 | 255 | 0.960237 | 26 | 0.835992 | 2,530.91 |

All candidates pass the overall recall gate, but none satisfies the combined
success criterion. Weight `1.50` is selected by the fixed minimum-FPR policy and
removes 272 false positives relative to frozen Context, but Fuzzer misses rise
from 261 to 270 and overall misses rise from 287 to 305. Weight `2.00` gives the
best Fuzzer result, recovering 17 Fuzzers and 11 attacks overall relative to
frozen Context, but adds 100 false positives and increases Analysis misses from
19 to 23. Weight `3.00` also modestly improves Fuzzers but degrades Analysis to
26 misses. No candidate reaches FPR below 2% or Fuzzer recall of 97%.

The weight `1.00` experiment intentionally calls the `sample_weight` training
path with an all-ones vector. Its scores are not identical to the historical
fit that omitted `sample_weight` (maximum absolute validation-score difference
`0.134414`), although their aggregate metrics are close. It is therefore the
proper within-experiment control, while frozen Context remains the operational
reference. This behavior was verified against scikit-learn 1.9.0: with
`sample_weight=None`, bootstrap indices use `randint`; with a supplied weight
vector, even an all-ones vector, they use `choice` with normalized weight
probabilities. The two modes therefore do not have to draw identical bootstrap
samples under the same seed. Reproducibility is required within a fixed mode.

The result is a Pareto trade-off rather than a successful correction: Fuzzer
weighting can move Fuzzer recall or FPR, but this grid does not improve both at
once and progressively harms the small Analysis category. The next justified
combination experiment is a meta-model trained only from January 22 OOF scores;
February 17 must remain clean validation for that meta-model.

Run the resumable experiment with:

```powershell
.\.venv\Scripts\python.exe src\train_v2_fuzzer_weighting.py
```

The five fits took 5,852 seconds in total; complete execution including data
loading, scoring, persistence, and checks took 5,901 seconds. All saved models
were reloaded and reproduced checked validation scores within `1e-12`.

Generated artifacts:

- `models/v2_context_fuzzer_weight_1p00.joblib`;
- `models/v2_context_fuzzer_weight_1p25.joblib`;
- `models/v2_context_fuzzer_weight_1p50.joblib`;
- `models/v2_context_fuzzer_weight_2p00.joblib`;
- `models/v2_context_fuzzer_weight_3p00.joblib`;
- `models/v2_fuzzer_weight_metrics.json`;
- `models/v2_fuzzer_weight_threshold_results.csv`;
- `models/v2_fuzzer_weight_comparison.csv`;
- `models/v2_fuzzer_weight_category_metrics.csv`;
- `models/v2_fuzzer_weight_validation_scores.parquet`.

The frozen Context model, prior metrics, and prior validation scores retained
their original SHA-256 hashes. February 18 was not loaded or evaluated.

### V2 purged temporal OOF stacking

The stacking experiment uses Jan 22 only for all model fitting. Base-model OOF
scores are chronological, not random-K-fold scores. Each base model trains on a
prefix and predicts a strictly later interval. Because Timestamp is deliberately
absent from the model-ready Parquet, 1,024 rows are purged after every training
boundary. This exceeds the previously measured maximum of 776 flows in one
timestamp second, so a tied timestamp group cannot cross a train/predict
boundary.

An initial equal-row five-block design was rejected during execution. Its first
353,184-row warm-up contained the complete positive class: all Jan 22 attacks
end by row 283,648, leaving zero attacks in all four OOF intervals. The stacker
therefore could not be fitted. No random split was substituted. Instead, the
Jan 22-only temporal boundaries were frozen at:

```text
warm-up: [0, 60,000)
fold 1:  train [0, 60,000)  -> predict [61,024, 120,000)
fold 2:  train [0, 120,000) -> predict [121,024, 180,000)
fold 3:  train [0, 180,000) -> predict [181,024, 240,000)
fold 4:  train [0, 240,000) -> predict [241,024, 1,765,922)
```

This produces 1,701,826 honest OOF rows: 1,691,100 normal and 10,726 attacks.
The four prediction folds contain 2,631, 3,174, 2,829, and 2,092 attacks
respectively. Every saved OOF row receives exactly one baseline score and one
Context score from models that used only earlier rows.

The meta-model input is intentionally limited to 17 features:

- baseline and Context scores, their difference, maximum, and minimum;
- `Protocol`;
- `dst_unique_sport_60s`, `src_dst_conn_60s`, and `src_dport_conn_60s`;
- source/destination 60-second connection and packet counts;
- source/destination mean packets per flow;
- source/destination UDP ratios.

The first stacker is a scaled Logistic Regression. Because it did not achieve
the strict validation target, the one predeclared fallback was also trained: a
HistGradientBoosting classifier with `max_depth=3`, 100 iterations, no early
stopping, and seed 42. Both stackers fit only Jan 22 OOF rows. February 17 is
used only for validation and threshold selection.

| Feb 17 metric | Frozen Context | Logistic stacker | Shallow HGB stacker |
|---|---:|---:|---:|
| Selected threshold | 0.10 | 0.01 | 0.04 |
| Recall | 0.985908 | **0.991555** | 0.985810 |
| Precision | 0.624269 | 0.606736 | **0.626232** |
| F1 | 0.764477 | 0.752819 | **0.765918** |
| ROC-AUC | **0.994684** | 0.994054 | 0.993627 |
| PR-AUC | **0.847746** | 0.816817 | 0.806098 |
| FPR | 0.025255 | 0.027353 | **0.025042** |
| False positives | 12,085 | 13,089 | **11,983** |
| Overall false negatives | 287 | **172** | 289 |
| Fuzzer false negatives | 261 | **106** | 255 |
| Analysis false negatives | 19 | 23 | **13** |
| FP per 100k benign | 2,525.47 | 2,735.29 | **2,504.16** |

The Logistic stacker proves that the two surfaces contain usable complementary
Fuzzer signal: it reduces Fuzzer misses by 155 and overall misses by 115. The
cost is 1,004 extra false positives, so it is not an operational improvement.
The shallow HGB stacker is selected by the fixed recall-gated minimum-FPR
policy. It removes only 102 false positives, recovers six Fuzzers and six
Analysis attacks, but adds two false negatives overall and materially lowers
PR-AUC. This marginal result does not justify a more complex production stack.

Neither stacker reaches the strict target of recall at least 98.5% with FPR
below 2%. The Logistic model does achieve Fuzzer FN below 200, but fails the FPR
gate; HGB fails both the FPR and Fuzzer targets. Therefore February 18 remains
locked. The evidence now favors evaluating incident aggregation over further
flow-classifier tuning: repeated false-positive flows may collapse into far
fewer source/destination/window incident candidates.

Run or resume the experiment with:

```powershell
.\.venv\Scripts\python.exe src\train_v2_temporal_stacking.py
```

The four OOF base-model pairs required 376.5 seconds of training and 14.3
seconds of inference. Both stackers together required 9.5 seconds of training.
All fold checkpoints and saved stackers emitted no warnings; reloaded stackers
reproduced checked scores exactly.

Generated artifacts:

- `models/v2_temporal_oof_scores.parquet`;
- `models/v2_temporal_oof_manifest.json`;
- `models/v2_stacker_logistic_regression.joblib`;
- `models/v2_stacker_hist_gradient_boosting.joblib`;
- `models/v2_stacking_metrics.json`;
- `models/v2_stacking_threshold_results.csv`;
- `models/v2_stacking_comparison.csv`;
- `models/v2_stacking_category_metrics.csv`;
- `models/v2_stacking_validation_scores.parquet`.

The frozen base models and prior validation scores retained their SHA-256
hashes. February 18 was not loaded or evaluated.

### V2 incident-level aggregation (Feb 17 only)

This evaluation-only stage keeps the 128-feature Context Random Forest and its
frozen threshold of `0.10` unchanged. It consumes the existing February 17
score vector; no classifier is loaded, fitted, tuned, calibrated, or replaced.
Raw `Flow ID`, source/destination IP, ports, protocol, and Timestamp are restored
only for incident grouping and diagnostics. They remain excluded from model
input.

The published 1.9-GB combined CSV is not chronologically sorted. A routing-only
stream therefore inspects Timestamp and materializes only rows dated February
17. Nonmatching rows remain opaque bytes: their features and labels never enter
DuckDB, a DataFrame, scoring, or evaluation. The 498,890 routed rows were
aligned to the frozen score order using all 128 model features plus both target
columns. All 498,890 rows matched, with zero feature-value mismatches and zero
score/target mismatches.

The unchanged flow-level reference is:

| Metric | Value |
|---|---:|
| Validation flows | 498,890 |
| Alert flows | 32,164 |
| True-positive attack flows | 20,079 |
| False-positive normal flows | 12,085 |
| False-negative attack flows | 287 |
| Recall | 98.5908% |
| FPR | 2.5255% |
| PR-AUC | 0.847746 |

Three deterministic keys were evaluated: A = `src_ip + dst_ip`, B =
`src_ip + dst_ip + dst_port`, and C =
`src_ip + dst_ip + dst_port + protocol`. Within each key, alerts are ordered by
Timestamp and frozen validation row. A new incident begins only when the gap
from the preceding alert for that key is greater than the selected window.
Same-second peers have zero gap. Interleaved traffic for another key does not
break a key's session.

An all-normal alert group is an FP incident. An attack-only group is a pure TP
incident. A mixed group is reported separately and counts as a justified TP
incident because it contains at least one real attack. Reference attack
incidents are independently formed from all actual attack flows under the same
key/window; a reference incident is detected when any member crossed the frozen
threshold.

| Policy | Window | Incidents | FP incidents | Mixed | Alert reduction | Attack-incident recall | Incidents/hour | FP incidents/hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 30s | 5,409 | 560 | 199 | 83.18% | 99.9392% | 1,498.68 | 155.16 |
| A | 60s | 2,709 | 304 | 207 | 91.58% | 100.0000% | 750.59 | 84.23 |
| A | 5m | 141 | 85 | 52 | 99.56% | 100.0000% | 39.07 | 23.55 |
| A | 15m | 80 | 40 | 40 | 99.75% | 100.0000% | 22.17 | 11.08 |
| B | 30s | 10,707 | 1,423 | 151 | 66.71% | 99.9464% | 2,966.61 | 394.27 |
| B | 60s | 9,542 | 1,301 | 157 | 70.33% | 99.9879% | 2,643.82 | 360.47 |
| **B** | **5m** | **5,408** | **870** | **230** | **83.19%** | **99.9780%** | **1,498.41** | **241.05** |
| B | 15m | 3,716 | 681 | 281 | 88.45% | 99.9680% | 1,029.60 | 188.69 |
| C | 30s | 10,877 | 1,436 | 148 | 66.18% | 99.9473% | 3,013.72 | 397.88 |
| C | 60s | 9,831 | 1,325 | 155 | 69.43% | 99.9883% | 2,723.90 | 367.12 |
| C | 5m | 6,007 | 929 | 222 | 81.32% | 99.9803% | 1,664.37 | 257.40 |
| C | 15m | 3,996 | 708 | 304 | 87.58% | 99.9704% | 1,107.18 | 196.17 |

The three operational candidates deliberately are not the three smallest raw
incident counts:

| Rank | Candidate | FP incidents | FP-flow/incident compression | Alert reduction | Attack-incident recall |
|---:|---|---:|---:|---:|---:|
| 1 | **B / 5m (recommended)** | 870 | 13.89x | 83.19% | 99.9780% |
| 2 | C / 5m | 929 | 13.01x | 81.32% | 99.9803% |
| 3 | A / 60s | 304 | 39.75x | 91.58% | 100.0000% |

Policy A at 5 or 15 minutes produces smaller headline counts, but it merges all
ports and protocols for a host pair. At 5 minutes its average incident contains
228 flows, its largest contains 1,717, and only 99 reference attack incidents
remain from 20,366 attack flows. This is an excessive cross-service merge risk,
so A is shortlisted only at 60 seconds. Between the service-aware B/C policies,
B is simpler, has 59 fewer FP incidents, and has essentially identical incident
recall; it is therefore recommended.

For B/5m, 12,085 FP flows become 870 all-normal incidents, while all 32,164
alerts become 5,408 analyst-facing incidents. Normal alerts compress 10.96x
when grouped separately and detected attack flows compress 4.41x. Of the normal
alerts, 8,869 occur inside 230 mixed incidents. This is not suppression: those
incidents remain visible to an analyst because they contain real attack flows.
The 3.609-hour capture contains 138,228.58 flows/hour and 8,911.75 pre-grouping
alerts/hour; B/5m yields 1,498.41 incidents/hour, including 241.05 pure FP
incidents/hour and 1,257.35 true-or-mixed security incidents/hour.

Recommended B/5m detection preservation is:

| Category | Attack flows | Detected flows | Reference incidents | Detected incidents | Flow recall | Incident recall |
|---|---:|---:|---:|---:|---:|---:|
| All attacks | 20,366 | 20,079 | 4,552 | 4,551 | 98.5908% | 99.9780% |
| Fuzzers | 6,413 | 6,152 | 155 | 155 | 95.9301% | 100.0000% |
| Analysis | 52 | 33 | 3 | 3 | 63.4615% | 100.0000% |
| Exploits | 7,268 | 7,263 | 2,959 | 2,958 | 99.9312% | 99.9662% |
| DoS | 1,037 | 1,037 | 707 | 707 | 100.0000% | 100.0000% |
| Reconnaissance | 3,870 | 3,870 | 890 | 890 | 100.0000% | 100.0000% |
| Generic | 1,074 | 1,072 | 713 | 713 | 99.8138% | 100.0000% |
| Backdoor | 98 | 98 | 75 | 75 | 100.0000% | 100.0000% |
| Shellcode | 489 | 489 | 351 | 351 | 100.0000% | 100.0000% |
| Worms | 65 | 65 | 41 | 41 | 100.0000% | 100.0000% |

FPs are concentrated but not reducible to one safe whitelist. At the raw flow
level, the top 20 `src/dst/dst_port/protocol` patterns account for 5,982 of
12,085 FPs (49.50%). TCP destination port 179 is especially common: the six
largest patterns alone contain 2,593 FP flows. UDP/520, TCP/80, TCP/860,
TCP/1723, and TCP/135 also appear among the top patterns. For B/5m all-normal
incidents, the top 20 keys account for 51.77% of FP-only incident flows. These
are diagnostics only; no entity, port, or protocol was suppressed or
whitelisted.

Run the analysis with:

```powershell
.\.venv\Scripts\python.exe src\analyze_incident_aggregation.py --memory-limit 10GB --threads 8
```

Generated artifacts:

- `models/v2_incident_aggregation_results.csv`;
- `models/v2_incident_aggregation_category_metrics.csv`;
- `models/v2_incident_aggregation_fp_patterns.csv`;
- `models/v2_incident_aggregation_manifest.json`.

The aggregation materially improves the flow detector's operational usability:
the recommended policy reduces analyst-facing alert objects by 83.19% and
retains 99.978% incident recall without changing the classifier. It does not
make the system production-ready: 241 pure FP incidents/hour remains high, and
flow-level weaknesses in Fuzzers and Analysis still exist. February 18 was not
materialized, scored, or evaluated and remains locked.

### V2 deterministic incident promotion

This stage adds an interpretable evidence gate after the frozen Context flow
detector and frozen Policy B / five-minute incident aggregation. It does not
change the 0.10 flow threshold, retrain a model, add a classifier, whitelist an
entity, or suppress a port.

Promotion selection uses only the existing Jan 22 causal temporal OOF scores.
The 1,701,826 OOF rows were aligned to their original Timestamp,
`src_ip`, `dst_ip`, and `dst_port`; all 1,765,922 Jan 22 feature rows matched
with zero feature-value mismatches. Only the 10,726 attack flows and 14,920
alert flows were materialized for incident evaluation. Feb 17 files were not
opened until the selected rule had been written to
`models/v2_incident_promotion_selected_policy.json` and hashed.

Every Policy B / 5m alert incident receives label-free, inference-time evidence:

- alert-flow count and duration;
- maximum, mean, median, p90, top-three mean, and standard deviation of scores;
- counts of flows with score at least 0.20, 0.30, 0.40, and 0.50.

Labels and attack categories are joined only to evaluate policies. The rules are
monotonic in maximum/count evidence, so their final result is equivalent to an
online gate that promotes as soon as its condition becomes true while an
incident accumulates chronologically.

The candidate grid was frozen in advance:

- Family A: `max_attack_score >= T`, with `T=0.10..0.90`, step 0.05;
- Family B: `max_attack_score >= T OR alert_flow_count >= N`, with four declared
  thresholds and four declared counts;
- Family C: `max_attack_score >= T OR score>=0.30 flow count >= N`, with three
  declared thresholds and three declared counts.

All 42 candidates are recorded in
`models/v2_incident_promotion_oof_results.csv`. Eight passed the required OOF
attack-incident recall of at least 99.9%. The minimum-FP candidate was the
simplest rule:

```text
promote incident when max_attack_score >= 0.25
```

OOF selection results:

| Metric | Ungated B/5m | Selected promotion | Change |
|---|---:|---:|---:|
| Reference attack incidents | 2,304 | 2,304 | — |
| Detected/promoted attack incidents | 2,304 | 2,302 | -2 |
| Incident recall | 100.0000% | 99.9132% | -0.0868 pp |
| Total analyst incidents | 3,269 | 2,730 | -539 / -16.49% |
| TP or mixed incidents | 2,303 | 2,301 | -2 |
| Pure FP incidents | 966 | 429 | -537 / -55.59% |
| Incident precision | 70.45% | 84.29% | +13.84 pp |
| Incidents/hour | 267.88 | 223.71 | -44.17 |
| FP incidents/hour | 79.16 | 35.16 | -44.01 |

The exact Family A / 0.25 rule was frozen before the single Feb 17 validation
pass. It was not modified afterward.

| Feb 17 metric | Before promotion | After promotion | Change |
|---|---:|---:|---:|
| Total incidents | 5,408 | 5,274 | -134 / -2.48% |
| Pure FP incidents | 870 | 739 | -131 / -15.06% |
| TP or mixed incidents | 4,538 | 4,535 | -3 |
| Reference attack incidents | 4,552 | 4,552 | — |
| Detected/promoted attack incidents | 4,551 | 4,548 | -3 |
| Missed attack incidents | 1 | 4 | +3 |
| Incident recall | 99.9780% | 99.9121% | -0.0659 pp |
| Incident precision | 83.91% | 85.99% | +2.08 pp |
| Incidents/hour | 1,498.41 | 1,461.28 | -37.13 |
| FP incidents/hour | 241.05 | 204.76 | -36.30 |

Feb 17 category preservation after promotion:

| Category | Reference incidents | Promoted incidents | Incident recall |
|---|---:|---:|---:|
| Fuzzers | 155 | 155 | 100.0000% |
| Analysis | 3 | 3 | 100.0000%* |
| Exploits | 2,959 | 2,957 | 99.9324% |
| DoS | 707 | 707 | 100.0000% |
| Reconnaissance | 890 | 890 | 100.0000% |
| Generic | 713 | 711 | 99.7195% |
| Backdoor | 75 | 75 | 100.0000% |
| Shellcode | 351 | 351 | 100.0000% |
| Worms | 41 | 41 | 100.0000% |

`*` Analysis contains only three reference incidents and is below the declared
sufficiency threshold of 20 incidents. It is reported but should not be treated
as a stable category estimate. No category result was used to modify the rule.

The promotion gate is a new **constraint-aware operational Pareto point**: it
reduces total and false-positive incidents while retaining the predeclared
99.9% incident-recall gate. It is not a strict unconstrained Pareto dominance,
because three additional Feb 17 reference incidents are missed. It is also not
a large workload breakthrough: total incidents fall only 2.48% and 204.76 pure
FP incidents/hour remains high. The substantially larger OOF reduction did not
fully generalize, which is an important result rather than a reason to retune on
Feb 17.

Run the experiment with:

```powershell
.\.venv\Scripts\python.exe src\analyze_incident_promotion.py --memory-limit 10GB --threads 8
```

Generated artifacts:

- `models/v2_incident_promotion_oof_results.csv`;
- `models/v2_incident_promotion_selected_policy.json`;
- `models/v2_incident_promotion_validation_metrics.json`;
- `models/v2_incident_promotion_category_metrics.csv`;
- `models/v2_incident_promotion_manifest.json`.

The selected-policy hash remained unchanged across validation. Frozen Context
model and score hashes also remained unchanged. Feb 18 features and labels were
not materialized, scored, or evaluated and remain locked.

### Final temporal holdout — Feb 18

February 18 was evaluated once as the locked temporal holdout after the entire
v2 pipeline was frozen. It was never used for model training, feature or
threshold selection, aggregation selection, or promotion selection. The
frozen system is the Context Random Forest with flow threshold `0.10`, Policy B
aggregation (`src_ip + dst_ip + dst_port`, five-minute gap), and promotion when
an incident's maximum attack score is at least `0.25`. No post-holdout tuning,
whitelisting, suppression, retraining, or deployment change was performed.

Before labels were loaded, the evaluator checked fixed SHA-256 snapshots for
the model, deterministic feature/preprocessing contract, feature builder,
threshold configuration, aggregation manifest, promotion policy/manifest, and
raw source. It then routed the 1,275,429 raw Feb 18 rows once without altering
them. The 00:00:00–08:29:24 inclusive capture contains 54,737 attack and
1,220,692 normal flows, or 150,222.29 flows/hour.

Frozen flow-level results at threshold `0.10`:

| Metric | Feb 17 validation | Feb 18 holdout | Change |
|---|---:|---:|---:|
| Recall | 98.5908% | 98.3686% | -0.2222 pp |
| Precision | 62.4269% | 67.5499% | +5.1230 pp |
| FPR | 2.5255% | 2.1190% | -0.4065 pp |
| PR-AUC | 0.847746 | 0.898915 | +0.051169 |
| TP / TN | — | 53,844 / 1,194,826 | — |
| FP / FN | 12,085 / 287 | 25,866 / 893 | raw counts are not time-normalized |
| FP flows/hour | 3,348.42 | 3,046.54 | -301.87/hour |

Feb 18 ROC-AUC is `0.996155`, F1 is `0.800970`, and FNR is
`1.6314%`. Flow recall remains high but misses the pre-existing 98.5% research
gate by 0.1314 percentage point. The lower FPR, higher PR-AUC, and lower
FP-flow rate show that this is not a broad ranking collapse.

The frozen incident pipeline reduces workload as follows. Flow recall and
incident recall have different denominators: the former counts attack flows,
while the latter counts reference attack incidents and regards an incident as
detected when at least one member flow survives the stage.

| Feb 18 stage | Volume | FP volume | Recall | Precision | Rate/hour | FP/hour |
|---|---:|---:|---:|---:|---:|---:|
| Raw flow detector | 79,710 alerts | 25,866 flows | 98.3686% | 67.55% | 9,388.39 | 3,046.54 |
| Policy B / 5m | 13,795 incidents | 1,721 incidents | 99.9917% | 87.52% | 1,624.80 | 202.70 |
| Policy B / 5m + promotion | 12,911 incidents | 844 incidents | 99.9339% | 93.46% | 1,520.68 | 99.41 |

Aggregation removes 65,915 analyst objects (82.69%) relative to flow alerts
and compresses 25,866 FP flows into 1,721 pure-FP incidents (15.03x). Promotion
then removes another 884 incidents (6.41%) and 877 pure-FP incidents (50.96%)
while changing missed reference attack incidents from one to eight. End to
end, the frozen incident pipeline reduces flow-alert volume by 83.80% and
FP-object volume by 96.74%.

Attack-category performance after the frozen promotion stage:

| Category | Attack flows | Flow FN | Flow recall | Reference incidents | Incident FN | Incident recall |
|---|---:|---:|---:|---:|---:|---:|
| Fuzzers | 17,573 | 793 | 95.4874% | 417 | 0 | 100.0000% |
| Analysis | 279 | 66 | 76.3441% | 10 | 0 | 100.0000%* |
| Exploits | 19,145 | 27 | 99.8590% | 7,884 | 4 | 99.9493% |
| DoS | 2,754 | 1 | 99.9637% | 1,879 | 1 | 99.9468% |
| Reconnaissance | 10,376 | 3 | 99.9711% | 2,356 | 1 | 99.9576% |
| Generic | 2,878 | 3 | 99.8958% | 1,938 | 2 | 99.8968% |
| Backdoor | 290 | 0 | 100.0000% | 226 | 1 | 99.5575% |
| Shellcode | 1,298 | 0 | 100.0000% | 937 | 0 | 100.0000% |
| Worms | 144 | 0 | 100.0000% | 92 | 0 | 100.0000% |

`*` Analysis has only ten reference incidents and is not a stable category
estimate. Fuzzers and Analysis remain the concentrated flow-level weaknesses,
but temporal aggregation recovers their incident-level signal.

Diagnostics did not alter the pipeline. Feb 17→Feb 18 protocol mix is stable
(Jensen–Shannon divergence `0.000014` bits), and PSI for the ten most important
numeric features is low (`0.00005`–`0.02220`). Incident evidence distributions
also show low PSI (`0.00228`–`0.02198`). A high-cardinality
`Protocol/Dst Port` service proxy has moderate divergence (`0.144845` bits)
and 31,796 Feb 18-only combinations, largely reflecting ephemeral-port
cardinality. CICFlowMeter has no explicit UNSW `service` or `state` fields, so
the report uses the documented service proxy and does not invent a state
feature. Score PSI is reported with the limitation that zero-heavy reference
deciles collapse to one effective bin; its median, p90, and p99 are unchanged,
while mean score decreases from `0.05342` to `0.05102`.

The Feb 18 top-20 pure-FP patterns contain 1,764 flows across 65 aggregated
incidents (6.82% of all FP flows and 3.78% of pure-FP incidents). TCP dominates
these patterns, with recurring destination-port families `445`, `860`, `179`,
`8009`, `8088`, and `8089`; UDP `53` and `514` also recur. This is structurally
similar to Feb 17's TCP/UDP service concentration, although only the UDP/53
identity pattern appears in both exact top-20 lists. No pattern was suppressed
or whitelisted.

The final classification is **Acceptable but operationally noisy**. Feb 18
does not show catastrophic temporal degradation: normalized FPR and pure-FP
incident rate improve, PR-AUC improves, aggregation retains 99.9917% incident
recall, and promotion retains 99.9339%. It is not classified as strong because
flow recall falls slightly below the established 98.5% gate, and 1,520.68
promoted incidents/hour remains too high for production Tier-1 use. Passing
this holdout alone does not make the system production-ready.

Run the immutable evaluator only for audit/reproduction with the project
environment:

```powershell
.\.venv\Scripts\python.exe src\evaluate_v2_feb18_holdout.py --memory-limit 10GB --threads 8
```

The original execution emitted a generic scikit-learn serialization warning
because the model was trained with 1.9.0 and evaluated under 1.8.0. A separate
Feb 17-only compatibility check reproduced all 498,890 frozen scores within
`2.22e-16` and produced zero threshold-class disagreements; Feb 18 was not
rescored.

Final artifacts:

- `models/v2_feb18_flow_metrics.json`;
- `models/v2_feb18_incident_metrics.json`;
- `models/v2_feb18_category_metrics.csv`;
- `models/v2_feb18_stage_comparison.csv`;
- `models/v2_feb18_drift_metrics.json`;
- `models/v2_feb18_fp_patterns.csv`;
- `models/v2_feb18_final_evaluation_manifest.json`.
