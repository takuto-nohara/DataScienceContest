# 本番提出ファイル（チーム：生体情報処理研究室、ファイル用チーム名 seitailab）

提出する2つのファイルです。どちらも `build_submission.m` で作り、提出前チェックアプリと同じ確認を通しています。

| ファイル | 中身 |
|---|---|
| `Pretreatment_MATLABcontest_Advanced_seitailab.m` | 前処理関数。公式の雛形の「ここから下を編集してよい」〜「ここまでを編集してよい」の間だけに、32列の特徴量の計算を入れたもの |
| `trainedModel_seitailab.mat` | 変数 `trainedModel`（ガウスSVM、`model/final_ergonomic32_svm/trained.mat` と同じ） |

`Pretreatment_MATLABcontest_Advanced.m` は、コピー元の前処理関数です（公式ライブスクリプトの提出作業３と同じく、これを `copyfile` で名前を変えています）。

## 確かめた結果（2026-11-26、MATLAB R2026a）

- 学習用2941窓を前処理関数に通した出力で、学習時と同じ判定になる（正解率0.9976）
- チェックアプリと同じ手順（ファイル名の形、変数名 `trainedModel`、検証用407窓）で 400/407（0.9828）。合格の基準は0.5以上
- 1試行だけ渡したときの出力が、まとめて渡したときと一致する（差の最大 4.5×10⁻¹⁴）

## 作り直す・確かめる

MATLAB でこのフォルダを現在のフォルダにして実行します。

```matlab
cd submission/seitailab
build_submission
```

チェックアプリ（`DScontest_submitModelVerifier`）で確かめるときも、このフォルダを指定します。アプリは前処理関数を、そのフォルダに移動してから呼び出します。

## 提出

締め切りは 2026年11月27日（金）8:59 です。上の2ファイルとスライド（PowerPoint か PDF）を、ポータルサイトにある提出フォームから送ります。
