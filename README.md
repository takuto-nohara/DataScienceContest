# MATLAB Data Science Contest 2026 Autumn — Advanced

Private development workspace for the human activity recognition contest.

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

See `official/README.md` and `data/README.md`. This repository does not contain
the organizer's datasets, live scripts, submission verifier, or a completed
registration. Only the environment can be verified before receiving those files.

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
