# proto_v3：モデルの開き方

`trained.mat` はMATLABのバイナリ形式です。テキストエディターで表示される不規則な文字は、文字コードの破損を示すものではありません。

方式：`rbf_svm` ／ 特徴量セット：`temporal32` ／ 特徴量数：32個

MATLABでは、リポジトリのフォルダで次を実行してください。`bundle` が学習済みモデル、`info` が概要です。

```matlab
addpath('matlab');
[bundle, info] = har.inspectModel('proto_v3');
```

VSCodeでは [model-info.json](model-info.json) をUTF-8で開くと、方式・設定・特徴量名を確認できます。詳しい学習記録は [manifest.json](manifest.json)、分類の仕組みは [日本語レポート](../../reports/classification-criteria-ja.md) を参照してください。
