# Test-aware model-development search log

The user explicitly requested analysis of the five baseline errors and repeated
improvement until all supplied 201 evaluation windows were classified correctly.
From this point, those labels are **development labels**, not an untouched test.
Neither fitting the test rows nor row-specific inference exceptions were used.
The validation labels were inspected repeatedly too.

Search stages, in order:

1. `har.searchRefinements(pwd)` — 48 temporal32 SVM settings.
2. `har.searchRefinements(pwd, {'stable26','gravity32','spectral32','distribution32','window32'})` — 240 settings.
3. `har.searchRefinements(pwd, {'shape32'})` — 48 settings.
4. `har.searchHierarchy(pwd)` — 168 combinations of static and dynamic specialists.
5. `har.searchAugmentation(pwd)` — 180 feature/gain/SVM combinations.
6. A general static-candidate majority-vote rule combines the best augmented
   primary with original v1 and v2. This corrects the newly remaining row 155
   without introducing any new development-set error.

CSV files contain every evaluated search configuration, including poor ones.
`searchRefinements` appends to search.csv; the other search stages replace their
own logs when explicitly rerun. Intermediate checkpoint MAT files stay local;
published, reproducible winners are in model/proto_v8, v9, and v10.

The saved versions use 2206 unique fitting windows, with no validation or
development-window fitting. v9/v10 have 6618 derived observations from gains
1, 0.8, and 1.2. The final 201/201 result uses a general feature-based rule and
holds when each window is evaluated individually or in shuffled order.

The 201/201 target was selected with these labels. New independently collected
subject/recording-held-out data are required to assess generalization.
