# MATLAB Data Science Contest 2026 Autumn — Advanced

Private development workspace for the human activity recognition contest.

## Prototype comparison

Seven versioned prototypes have been trained and evaluated on the supplied
prestudy dataset. See [the experiment report](evaluation/latest/README.md),
the sortable [HTML comparison](evaluation/latest/report.html), and the
[model registry](model/README.md).

```powershell
# Train every model/proto_*/spec.json and compare every saved model
.\scripts\run_prototypes.ps1

# Evaluate saved models without retraining
.\scripts\run_prototypes.ps1 -Action evaluate

# Compare any chosen number of versions (example: three)
.\scripts\run_prototypes.ps1 -Action evaluate -Models proto_v1,proto_v3,proto_v6 -OutputName subset_check

# MATLAB checks; Python stdlib independently verifies report calculations
.\scripts\run_prototypes.ps1 -Action test
python -I -B tests\check_report.py
```

Models live in `model/proto_vN/`. All read the same `data/prestudy` directory.
Actual trained models, specifications, manifests, and evaluation reports are
tracked; organizer data remain local under `data` with tracked provenance.

The fixed protocol uses 2206 fitting windows and 735 validation windows from
the organizer training file, plus all 201 windows from its separate test file.
Predefined prototypes are selected by validation accuracy; external test scores
are reported descriptively. There are no subject IDs, so this random window
split cannot establish performance on unseen people and may include related
or overlapping windows. No test-driven hyperparameter tuning was performed.
All configurations were fixed before the first external-test comparison.

The evaluator writes CSV/JSON summaries, per-class precision/recall/F1, confusion
matrices, every test prediction, a PNG chart, and a standalone sortable HTML
report. `har.evaluate` never trains or tunes models, and checks model artifact
hashes, spec consistency, dataset hashes, class order, and split compatibility.
Models trained on different datasets or protocols cannot silently be compared.

MATLAB R2026a Update 5 with Statistics and Machine Learning Toolbox is required
for these prototypes; Python 3 is optional and used only for the independent
report checks. No third-party Python dependencies are required.

## Environment

- MATLAB **R2026a Update 5**
- Deep Learning Toolbox
- Parallel Computing Toolbox
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox (optional in the rules, included in this setup)
- Installation: `C:\Users\nohataku\MATLAB\R2026a`

Installation alone does not prove readiness. The university MathWorks sign-in
and a successful `ENVIRONMENT_CHECK_PASSED` result are required before claiming
that all licensed functionality works. Actual contest development also requires
the organizer's distributed files.

Verified on 2026-11-01: R2026a Update 5, all five product license checkouts,
synthetic-data classification, deep learning operations, signal processing,
a two-worker process pool, and an NVIDIA RTX 4070 GPU computation all passed.
The launch scripts explicitly use online licensing for the university account.
An initial post-installation failure was resolved by running MathWorks'
`registerWithOS.exe` with administrator privileges.
The toolbox lookup cache was then rebuilt using
`restoredefaultpath; rehash toolboxcache; savepath`, followed by a fresh-session
verification.

Launch from this directory:

```powershell
.\scripts\start_matlab.ps1
```

After signing in to the university's MathWorks license, verify real toolbox
license checkouts, synthetic-data classification, deep learning operations,
signal processing, and a two-worker parallel pool:

```powershell
.\scripts\verify_environment.ps1
```

The log is written to `artifacts/environment-check.log`. GPU checks are optional;
CPU development does not depend on a supported GPU. The verification script
does not train a contest model or submit anything.

## Organizer materials

See `official/README.md` and `data/README.md`. The supplied prestudy ZIP and MAT
files are available locally in `data/prestudy` and are excluded from Git.
The official Advanced live script, preprocessing template, and submission
verifier have not been supplied. No registration or submission forms were sent.

## Development constraints

- Six input signals: total_acc_x/y/z and body_gyro_x/y/z.
- Each trial contains 128 samples at 50 Hz.
- Pretreatment output: real double matrix with **N rows and exactly 32 columns**;
  pad unused feature columns with zero.
- Preserve trial order; do not remove trials.
- Compute each trial's features independently. A single trial must give the
  same features whether processed alone or in a batch.
- Do not normalize across input trials. Use training-fitted transformations
  in Classification Learner where needed.
- Only edit the marked region of the official Pretreatment function. Preserve
  its function name, six inputs, and output name `ActivityDataTrain`.
- Preprocessing may use only MATLAB, the four listed toolboxes, and local
  functions in the same file.
- Keep the test set independent from feature/model selection and final tuning.
- Run the organizer's submission verifier once it is supplied.
- Let the official live script generate submission filenames.

## Dates (Japan Standard Time)

- Team entry: 2026-11-16 23:59
- Official development period listed: 2026-11-18 through 2026-11-26
- Submission deadline: 2026-11-27 08:59
- Contest: 2026-11-27 10:30–12:00 (may run longer)

Source: [Organizer portal](https://docs.google.com/document/d/e/2PACX-1vT_0dNx2LIx2SMLat-mRBv63TXpRC4fRMwKg8UbgWHzfhe5UMKVvwk8DqjsMlBPEpUBBCluLwd6rpYJ/pub),
read on 2026-11-01; page marked updated 2026-10-28 21:45.

Installation uses the signed official [MATLAB Package Manager](https://www.mathworks.com/help/install/ug/mpminstall.html),
with `--release=R2026aU5` and the five product names above.
To reproduce the installation, run `scripts/install_matlab.ps1`.
Credentials and license files belong outside this repository.
