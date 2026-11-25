# 本番発表のスライド

Claude Design（Slides）で作ったスライドの元ファイルと、PowerPoint への書き出しです。

| フォルダ | 内容 | PowerPoint |
|---|---|---|
| `final_seitailab/` | 本番提出スライド（8枚）。分類割合、特徴量の設計と試行錯誤、特徴量を選んだ理由、モデルを選んだ理由、混同行列の比較 | `../final_seitailab.pptx` |
| `production_data_results/` | 本番配布のデータの結果と試行錯誤（2枚） | `../production_data_results.pptx` |

各スライドは `project/slides/<id>.html`、共通の見た目は `project/deck.css`、順番は `project/deck.json` にあります。`<aside>` は話す内容（スピーカーノート）です。図は `../images/` の PNG で、`make_slide_figures.m` で作ります。

## PowerPoint を作り直す

`tools/` の4つで、ブラウザに表示したときの配置をそのまま PowerPoint に移します（Python と python-pptx が必要です）。

1. `mkpreview.py <デッキのフォルダ> <出力.html> img`：1枚 1920×1080 のプレビューページを作る（`img/` に図の PNG を置く）
2. `serve.py 8765`：プレビューを配信する。プレビューに `extract.js` を読み込ませると、各要素の位置と見た目を `layout*.json` に保存する
3. `build_pptx.py <layout.json> <デッキのフォルダ> <図のフォルダ> <出力.pptx>`：PowerPoint を作る。1920px を 13.33 インチとするので、1px は 0.5pt

Google スライドには、大学アカウントのドライブに PowerPoint を上げ、「Google スライドとして保存」で変換しています。
