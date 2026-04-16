---
title: "鏡面ワークを斜めから撮って射影変換で補正する：曇り面積の定量化"
emoji: "🔮"
type: "tech"
topics: ["python", "opencv", "画像処理", "製造業", "DX"]
published: true
---

# 鏡面ワークを斜めから撮って射影変換で補正する：曇り面積の定量化

## 現場でこんな状況ありませんか？

成膜や研磨の工程を経た**鏡面ワーク**の表面に、肉眼では「なんとなく白く曇っている」ような模様が現れることがあります。これはワーク表面の微小な凹凸による光の散乱です。問題は、**その模様がどのくらいの面積なのか、誰も数値で把握していない**ことです。

「前回よりひどい気がする」「だいたいOKでしょ」——こういった判断が続くと、検査の基準が人によってばらつき、やがてロットによって合否が変わる属人化した検査になっていきます。状態の悪い部材をそのまま使い続けると、間接的にワークの品質低下を引き起こす可能性もあります。

ならばカメラで撮って定量化しよう。当然の発想ですが、鏡面ワークには厄介な問題があります。**正面にカメラを置くと、カメラ自身が映り込んでしまうのです。**

映り込みを避けるには斜めから撮るしかありません。しかし斜めから撮ると、円形のワークが楕円に見えたり、奥行き方向に面積が圧縮されたりして、画像上の面積が実際の面積と一致しなくなります。

この記事では、**射影変換（ホモグラフィ）**を使って斜め撮像画像を正面補正し、曇り模様の面積を定量化する方法を紹介します。

---

## アイデアの核心

斜めから撮影すると、カメラから遠い側は近い側より小さく写ります（遠近法による歪み）。この歪みがあると、画像上のピクセル数を面積に換算しても実際の値と一致しません。

使う技術は**射影変換**（`cv2.getPerspectiveTransform` / `cv2.warpPerspective`）です。

```
斜め撮像画像のワーク外周4点（対応点）を指定する
　↓
「このワーク、正面から見たらこうなるはず」という変換行列を計算する
　↓
補正後の正面画像から曇り面積を算出する
```

難しい数学は不要です。「斜めに見えているワークの外周4点がどこにあるか」さえ指定すれば、変換行列の計算はOpenCVがやってくれます。

---

## 実装

`opencv-python` と `numpy` を使います（`pip install opencv-python numpy`）。

コードは2つのブロックで構成しています。

### ブロック①：射影変換で正面補正

冒頭のパラメータブロックがこの実装の肝です。`ANGLED_CORNERS` に斜め撮像画像上の楕円外周4点（上・右・下・左）の座標を、`FRONTAL_CORNERS` に補正後の正面画像での対応点を指定します。

実運用では、撮影フレーム内に固定された治具の四隅を使うと安定します。治具は動かないため、一度座標を設定すれば再調整は不要です。GUIツールで画像をクリックして座標を取得し、JSON で保存して固定するのが定番です。

![corner_selection.png](https://storage.googleapis.com/zenn-user-upload/0ec79c0f73f6-20260416.png)

```python
import csv
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# 設定パラメータ
# ============================================================
IMG_W, IMG_H = 640, 480

# 斜め撮像画像上の対応点（楕円外周4点：上・右・下・左）
# ※ 実運用では GUI でクリックして取得し、JSON で保存して固定する
ANGLED_CORNERS = np.float32([
    [379.3,  78.6],   # 上
    [512.0, 240.0],   # 右
    [379.3, 401.4],   # 下
    [257.6, 240.0],   # 左
])

# 補正後の正面画像上での対応点
FRONTAL_CORNERS = np.float32([
    [320,  50],   # 上
    [510, 240],   # 右
    [320, 430],   # 下
    [130, 240],   # 左
])

# ワーク領域（円形マスク用）
WORK_CENTER = (IMG_W // 2, IMG_H // 2)
WORK_RADIUS = 190  # [px]

# スケールキャリブレーション
# 例：ワーク直径 300 mm = 380 px の場合
# 実際の値はリファレンスゲージで計測して設定する
PX_PER_MM   = 380 / 300
MM2_PER_PX2 = (1.0 / PX_PER_MM) ** 2

# 曇り検出の閾値（0〜255）
# 鏡面部と曇り部の輝度分布を確認してから設定する
FOG_THRESHOLD = 110


def correct_perspective(img: np.ndarray) -> np.ndarray:
    """射影変換で斜め画像を正面補正する"""
    M = cv2.getPerspectiveTransform(ANGLED_CORNERS, FRONTAL_CORNERS)
    return cv2.warpPerspective(img, M, (IMG_W, IMG_H))
```

`getPerspectiveTransform` は「変換前の4点と変換後の4点」を渡すだけで変換行列を返します。`warpPerspective` でその行列を画像全体に適用すると、斜め画像が正面から見た状態に補正されます。

### ブロック②：曇り面積の算出

補正後の画像に対して、輝度閾値で曇り領域を二値化し、白ピクセル数をカウントします。円形のワーク領域だけをマスクして検査対象を限定しているのがポイントです。

```python
def measure_fog_area(corrected: np.ndarray) -> tuple[np.ndarray, int, float]:
    """補正画像から曇り領域を検出し面積を算出する"""
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    # ワーク領域（円形）だけを検査対象にする
    work_mask = np.zeros_like(gray)
    cv2.circle(work_mask, WORK_CENTER, WORK_RADIUS, 255, -1)

    # 固定閾値で曇り領域を抽出する
    # 照明条件を固定した環境では鏡面部（暗い）と曇り部（明るい）の輝度差が大きく、
    # 単純な閾値で分離できる
    _, binary_full = cv2.threshold(gray, FOG_THRESHOLD, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary_full, work_mask)

    fog_px  = int(np.count_nonzero(binary))
    fog_mm2 = fog_px * MM2_PER_PX2

    return binary, fog_px, fog_mm2


def save_log(log_path: Path, fog_px: int, fog_mm2: float) -> None:
    """計測結果を CSV に追記する"""
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "fog_px", "fog_mm2"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            fog_px,
            f"{fog_mm2:.1f}",
        ])


def main():
    input_path  = Path(__file__).parent / "samples" / "angled.png"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # ① 射影変換で正面補正
    angled = cv2.imread(str(input_path))
    corrected = correct_perspective(angled)
    cv2.imwrite(str(results_dir / "corrected.png"), corrected)

    # ② 曇り面積の算出
    binary, fog_px, fog_mm2 = measure_fog_area(corrected)
    cv2.imwrite(str(results_dir / "binary.png"), binary)

    save_log(results_dir / "log.csv", fog_px, fog_mm2)

    print(f"曇り面積: {fog_px} px²  →  {fog_mm2:.1f} mm²")
    print("結果を results/ に保存しました")


if __name__ == "__main__":
    main()
```

---

## 実行結果

サンプル画像（斜め撮像を再現した `angled.png`）に対して実行すると、次のように出力されます。

```
曇り面積: 16577 px²  →  10331.9 mm²
結果を results/ に保存しました
```

入力画像（`angled.png`）は斜め撮像で楕円に歪んだワークです。

![angled.png](https://storage.googleapis.com/zenn-user-upload/c32aabb3112e-20260416.png)

`results/corrected.png`（左）で射影変換後の正円に補正されたワーク、`results/binary.png`（右）で検出された曇り領域（白）を確認できます。

![corrected.png](https://storage.googleapis.com/zenn-user-upload/1a7572b32a6f-20260416.png) ![binary.png](https://storage.googleapis.com/zenn-user-upload/0f2fe3834b7c-20260416.png)

---

## 現場導入のポイントと想定課題

### 撮影環境：照明と遮光が精度の9割を決める

鏡面ワークの検査では、外光の写り込みが最大の敵です。照明条件が変わるたびに閾値が狂い、カメラが安価でも高価でも結果は同じです。**暗幕ブースを作って外光を遮断**し、照明条件を固定することが安定検査の要です。

カメラ自体は安価な USB カメラで十分です。お金をかけるべきは照明環境の整備です。

### 閾値の設定：輝度ヒストグラムで決める

固定閾値の最適値は現場ごとに異なります。次のコードで輝度ヒストグラムを確認しながら決定してください。

```python
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("corrected.png", cv2.IMREAD_GRAYSCALE)
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.axvline(110, color='red', label='FOG_THRESHOLD=110')
plt.legend()
plt.show()
```

鏡面部（暗い）と曇り部（明るい）の間にある谷を閾値に設定します。大津の二値化は背景の黒ピクセルに引っ張られて誤動作するため、固定閾値を推奨します。

### ワーク位置ずれへの対応

ワークのセット位置がズレると対応点（`ANGLED_CORNERS`）もズレ、補正精度が落ちます。

- **まず試すこと**：治具でワークの位置を固定し、ズレの発生自体を防ぐ
- **それでもダメなら**：テンプレートマッチングでワーク位置を自動検出し、`ANGLED_CORNERS` を動的に補正する

### ログ保存：3枚セットで後追いできるようにする

「なぜこのロットだけ数値が高い？」という後追い確認に備え、補正前・補正後・二値化画像の3枚をタイムスタンプ付きで保存します。

```python
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
cv2.imwrite(f"results/{ts}_angled.png", angled)
cv2.imwrite(f"results/{ts}_corrected.png", corrected)
cv2.imwrite(f"results/{ts}_binary.png", binary)
```

---

## まとめ

| 項目 | 内容 |
|---|---|
| 使用技術 | OpenCV 射影変換（ホモグラフィ） |
| 必要なもの | カメラ（斜め固定）、暗幕ブース、Python 環境 |
| 出力 | 曇り面積（mm²）＋補正画像＋二値化画像＋CSV ログ |
| 設備改造 | 不要 |

「正面から撮れない」という制約は、一見すると大きな障壁に見えます。しかし射影変換を使えば、斜め撮像でも再現性のある面積計測が実現できます。変換行列の計算はOpenCVに任せてしまえばよく、実装の核心は「対応点を正確に指定すること」だけです。

目視で「なんとなく」判断していた曇り模様に数値を与えること。それが、検査の属人化を崩す最初の一手になります。

---

## GitHubリポジトリ

サンプルコード・サンプル画像一式を公開しています。

🔗 [github.com/factory-dx-eng/manufacturing-python/mirror_surface_homography](https://github.com/factory-dx-eng/manufacturing-python/tree/main/mirror_surface_homography)

---
