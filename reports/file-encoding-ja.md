# ファイル形式と文字コード

## テキストとして読むファイル

`.m`、`.md`、`.json`、`.csv`、`.html`、`.py`、`.ps1` はUTF-8・BOMなしで保存します。VSCodeのワークスペース設定と `.editorconfig` にもUTF-8を指定しています。MATLABのスクリプト・関数は、R2020a以降ではUTF-8・BOMなしが標準です。[MathWorksの保存仕様](https://www.mathworks.com/help/matlab/matlab_prog/save-and-back-up-code.html)

CSV・JSON・レポートの書き出し処理も、UTF-8を明示するようにしています。既存テキストの監査では、不正なUTF-8やUnicodeの置換文字は見つかりませんでした。

## `.mat` はMATLABで読み込む

`trained.mat` は学習済みモデルを保存したバイナリファイルです。UTF-8の文章ではないため、VSCodeのテキストエディターでは不規則な文字として見えます。文字コードを変更して上書きするとモデルを壊す可能性があるため、バイナリ形式を維持しています。[MathWorksのMAT形式仕様](https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf)

各 `model/proto_vN/` には、次のファイルを用意しています。

| ファイル | 読み方 |
|---|---|
| `trained.mat` | MATLABの `load` で読み込む学習済みモデル |
| `model-info.json` | VSCode・MATLABのエディターで読めるUTF-8の概要。方式、設定、特徴量名を記載 |
| `README.md` | そのモデルの開き方を日本語で説明 |
| `manifest.json` | 学習条件、使用した行、モデルのハッシュなどの記録 |

例えば `proto_v10` をMATLABで確認する場合は、リポジトリのフォルダで次を実行します。

```matlab
addpath('matlab');
[bundle, info] = har.inspectModel('proto_v10');
```

`bundle` が学習済みモデル、`info` が読みやすい概要です。VSCodeでは [model-info.json](../model/proto_v10/model-info.json) を開いてください。モデルの実行には `bundle` 内の分類器を使うため、概要JSONだけで `.mat` を置き換えることはできません。

管理している10個の `trained.mat` は、MATLABでの読み込みとハッシュの一致を確認しています。

## コマンド実行時の表示崩れへの対応

このPCの日本語Windows環境では、MATLABのファイル書き出しはUTF-8ですが、リダイレクトされた標準出力はCP932でした。UTF-8として解釈する実行環境では、ファイルが正常でもコマンド結果の日本語が崩れて表示されます。

PowerShellの実行スクリプトは、子プロセスの出力をこの環境の文字コードで読み取り、文字列として表示する方式に変更しました。保存する実行ログはUTF-8・BOMなしです。Windows全体の言語設定やシステムロケールは変更していません。

通常は次のスクリプトを使ってください。

```powershell
.\scripts\verify_environment.ps1
.\scripts\run_prototypes.ps1 -Action evaluate
```

MATLABを直接起動したコマンドや、別のアプリ独自の出力経路には、この変換処理は入りません。

## 再確認するコマンド

```powershell
# UTF-8のテキスト、MATバイナリのハッシュ、概要JSONを確認
python -I -B tests\check_text_encoding.py

# 引数の受け渡し、日本語表示、UTF-8ログ、MATLABでの読み込みを確認
.\tests\check_native_utf8.ps1
```

MATLABで概要ファイルを再出力するときは `run('scripts/export_model_info.m')` を実行します。モデルの再学習やバイナリの書き換えは行いません。
