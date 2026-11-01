# Versioned prototypes

Each `proto_vN/` directory contains:

- `spec.json`: stage, method, feature set, fitting fraction, fixed seed.
- `trained.mat`: actual fitted MATLAB estimator and its training-only scaler.
- `manifest.json`: feature names, actual fit-row indices, parameters, training
  duration, dataset fingerprints, MATLAB version, and model SHA-256.

These are prototypes for testing the evaluator, not official submission files.
No prototype is refitted on the validation or external test rows.

| ID | Method | Features | Data used for fitting |
|---|---|---|---|
| proto_v1 | Decision tree | stats12 | 100% of common fitting partition |
| proto_v2 | Linear SVM (one-vs-one ECOC) | stats12 | 100% |
| proto_v3 | Gaussian/RBF SVM (one-vs-one ECOC) | temporal32 | 100% |
| proto_v4 | 100 bagged trees | temporal32 | 100% |
| proto_v5 | Weighted 5-nearest neighbors | temporal32 | Stratified 80% |
| proto_v6 | 24 k-means clusters + training-only majority class mapping | temporal32 | 100% |
| proto_v7 | Gaussian/RBF SVM | accel20 | Stratified 60%; gyroscope excluded |
| proto_v8 | Gaussian/RBF SVM, C=10, scale=8 | spectral32 | 100%; development-aware tuning |
| proto_v9 | Gaussian/RBF SVM, C=1, scale=4 | spectral32 | 100%; amplitude gains 1/0.8/1.2, 6618 fitting observations |
| proto_v10 | v9-style primary + static-candidate majority vote | spectral32 | Same augmentation; embeds original v1 tree and v2 linear SVM |

100% here means 2206 fitting windows, not all 2941 organizer training windows.
735 windows remain untouched for validation. Fractions are applied per class
and rounded down, giving 1762 and 1320 fitting rows for v5 and v7.

## Feature sets

All feature sets operate independently on each trial and use no batch-wide
statistics. Standardization is a separate learned transformation fitted only
on the saved model's fitting rows.

- `stats12`: mean and standard deviation for each of the six channels.
- `temporal32`: stats12 + mean/std/max/RMS temporal derivative of each sensor's
  vector magnitude (8), dominant nonzero FFT frequency per channel (6), and
  within-sensor xy/yz/zx correlations (6). Constant signals get frequency and
  correlation values of zero. Frequency extraction removes DC.
- `accel20`: accelerometer mean/std (6), magnitude statistics (4), dominant
  frequencies (3), correlations (3), maxima (3), magnitude spectral entropy (1).

Features have 12, 32, or 20 columns respectively. They are not padded into an
official preprocessing function. The largest set respects the contest's
32-feature budget, but the organizer verifier and official function contract
still need a separate integration step.

Refinement feature sets: `stable26` drops the six dominant-frequency bins;
`spectral32` replaces them with per-axis 0.5-3 Hz power fractions;
`gravity32`, `distribution32`, `window32`, and `shape32` test posture direction,
within-window quantiles, temporal changes, and gravity-projected impact shape.
All have at most 32 columns. Unsuccessful feature families remain implemented
and logged to make the investigation reproducible.

`proto_v10` uses one spectral32 vector; its first 12 columns are exactly stats12.
The primary SVM sees all 32, while the embedded v1/v2 models use that 12-column
subset with their original training-only scalers. When the primary predicts a
static class, a strict majority across the three six-class predictions is used;
ties retain the primary. Dynamic primary predictions are retained unchanged.
This is a general inference rule, not a list of special test rows.

The primary training augmentation preserves each window's estimated gravity
vector and scales its motion component and gyroscope signal. All 6618 derived
observations originate from the 2206 fitting windows, never the 735 validation
or 201 development windows. `fitRows` records unique source rows;
`augmentedRows`/`fittingObservations` record derived observations.

Versions v8-v10 have `evaluationRole=development_tuned`. The old test labels
guided their design. A new, independently collected subject/recording-held-out
dataset is required before claiming improved real-world generalization.

## Add another version

1. Create `model/proto_v8/spec.json` with a unique matching ID and
   `"stage":"prototype"`. Copy an existing spec, change features, method, seed,
   or training fraction, and document the hypothesis.
2. Run `scripts/run_prototypes.ps1 -Action train -Models proto_v8`.
3. Run `scripts/run_prototypes.ps1 -Action evaluate` to compare **all** versions.

Supported methods are tree, linear_svm, rbf_svm, bagged_trees, knn, and kmeans.
Adding a new method requires updating har.train/har.predict, not the comparison
loop. Changing feature definitions requires a new feature-set identifier and
retraining affected versions; feature names are checked at prediction time.

## Predict using a saved version

```matlab
addpath('matlab');
data = har.loadData(pwd);
saved = load('model/proto_v3/trained.mat', 'bundle');
predictedLabels = har.predict(saved.bundle, data.testRaw);
```

The input array is N x 128 x 6, ordered as acceleration x/y/z, then gyroscope
x/y/z. Predictions are activity-name strings. Saved models use only the
toolboxes installed for the contest. No Python packages are needed.
