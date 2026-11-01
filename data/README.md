# Shared organizer data

All prototypes reference this directory. No model directory contains a dataset copy.

Downloaded on 2026-11-01 from the user-provided organizer folder:
https://drive.google.com/drive/folders/1-RZWVPlhsY_WEvZakkYgf5T6YoOQt7Pl

```text
data/
  manifest.json                          # tracked provenance and SHA-256 hashes
  DScontest_Advanced_PreStudy.zip         # local only
  prestudy/
    Excercise1_HumanActivityClustering_Work_ja.mlx
    Excercise2_HumanActivityClassification_ML_ja.mlx
    Rawdata Files/
      rawSensorData_train.mat             # 2941 x 128 per channel
      rawSensorData_test.mat              # 201 x 128 per channel
      HumanActivityPredictions.mp4
```

Retain original filenames and variable names. The loader reads the six raw
channels and activity labels; the supplied 12-column ActivityDataTest feature
matrix is intentionally ignored so train and test use identical extraction.

Class order: Walking, WalkingUpstairs, WalkingDownstairs, Sitting, Standing,
Laying. Sampling rate: 50 Hz. Window length: 128 samples (~2.56 seconds).
There are no subject or recording identifiers in these MAT files.

Only this README and manifest.json are tracked in Git. Raw datasets, archives,
videos, and organizer live scripts remain local. To restore on another PC,
download the same ZIP through the authorized Drive folder, verify its SHA-256
against manifest.json, and extract it into data/prestudy. The manifest hashes
also pin each MAT file; the evaluator rejects changed datasets.
