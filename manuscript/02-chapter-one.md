# 第1章：環境構築とビルド

この章ではローカルと CI の双方で再現可能なビルド手順を示す。
Pandoc を利用して Markdown から EPUB3 を生成する。

## ローカルのセットアップ

Homebrew を利用して Pandoc をインストールする。
`brew install pandoc` を実行してコマンドが使えることを確認する。

## コマンドの基本形

以下のコマンドは本テンプレートの標準的なビルド定義である。
必要に応じてハイライトスタイルや CSS を調整する。

```
pandoc \
  --from=gfm \
  --to=epub3 \
  --standalone \
  --metadata-file=metadata.yml \
  --output=build/book.epub \
  --toc \
  --epub-stylesheet=epub.css \
  --highlight-style=kate \
  manuscript/*.md
```

## 画像の埋め込み

次の例のように相対パスで画像を参照する。
`![図1](images/ch01-diagram.png)` と書くと EPUB に同梱される。

TODO [編集者A]: ここに図版の差し替え予定をメモする。

