# work-position-measurement

搬送ライン上のワーク位置ずれをテンプレートマッチングで定量化するPythonサンプルです。

## 概要

カメラ1台とPythonだけで、ワークのXY方向の位置ずれをmm単位で計測します。
センサーの後付けや設備改造は不要です。

```
基準画像（正常位置のワーク）を1枚登録
　↓
毎回の検査画像とテンプレートマッチングで比較
　↓
ずれ量（px）をmm換算して出力
```

## 使い方

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. サンプルの実行

```bash
python main.py
```

### 3. サンプル画像を追加したい場合

`generate_samples.py` の `OFFSET_CASES` にずれ量を追加して実行してください。

```python
OFFSET_CASES = [
    ( 3.0,  0.0),
    ( 0.0, -2.0),
    ( 5.0,  5.0),   # ← 追加
]
```

```bash
python generate_samples.py
```

ファイル名にずれ量が記載されるので、計測結果との照合が簡単にできます。

```
test_dx+3.0_dy+0.0.png     3.00     0.00   1.000    OK
test_dx+0.0_dy-2.0.png     0.00    -2.00   1.000    OK
```

### 4. 実画像への適用

`main.py` の `PX_PER_MM` を実環境に合わせて変更してください。

```python
PX_PER_MM = 10.0   # 求め方：ピクセル数 ÷ 既知寸法(mm)
                    # 例：100mmの治具が200pxで写っていれば 200/100=2.0
```

## 出力例

```
基準画像: samples/reference.png
検査画像: 4件

============================================================
ファイル名                  dX(mm)   dY(mm)     スコア    判定
------------------------------------------------------------
test_dx+0.0_dy-2.0.png     0.00    -2.00   1.000    OK
test_dx+3.0_dy+0.0.png     3.00     0.00   1.000    OK
test_dx+4.0_dy+3.0.png     4.00     3.00   1.000    OK
test_dx-3.0_dy+2.5.png    -3.00     2.50   1.000    OK
============================================================

ログ保存: offset_log.csv
```

## ディレクトリ構成

```
work-position-measurement/
├── main.py                          # メインコード
├── generate_samples.py              # サンプル画像生成（動作確認・追加用）
├── requirements.txt                 # 依存パッケージ
├── .gitignore
├── README.md
├── samples/
│   ├── reference.png                # 基準画像（ずれなし）
│   ├── test_dx+3.0_dy+0.0.png      # 検査画像（ファイル名にずれ量を記載）
│   ├── test_dx+0.0_dy-2.0.png
│   ├── test_dx+4.0_dy+3.0.png
│   └── test_dx-3.0_dy+2.5.png
└── articles/
    ├── work-position-measurement.md # Zenn連携記事
    └── images/                      # 記事掲載用画像
        ├── 01_sample_images.png
        ├── 02_matching_process.png
        └── 03_measurement_results.png
```

## 関連記事

- [Zenn：搬送ライン上のワーク位置ずれをPythonで定量化する](https://zenn.dev/dx_bansou)

## ライセンス

MIT
