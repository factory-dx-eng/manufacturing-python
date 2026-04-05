---
title: "輪郭検出から始める外観検査 ― 外側・内側の2パターンをOpenCVで実装する"
emoji: "🔍"
type: "tech"
topics: ["python", "opencv", "画像処理", "製造業", "DX"]
published: true
---

# 輪郭検出から始める外観検査

## センサーで「点」を見るより、カメラで「面」を見る方がいいケースがある

透明な素材の中に薄膜を挟み込む工程がある。薄膜の有無を確認するために、レーザセンサーで検査していた。しかし、センサーの照射位置がほんの少しずれると誤検出が出る。ワーク自体のばらつきも重なって、誤検出を完全になくせない。

やがて「疑わしいものは全部目視で確認して」という運用になり、作業者に負担が集中した。

ある日、現場から「これ、カメラでできないですかね？」と声がかかった。

確かに、と思った。センサーは「1点」を見るが、カメラは「面」で捉える。照射位置のずれに引っ張られず、ワーク全体の状態を画像として評価できる。透明同士の重なりはコントラストが低く、単純な輝度判定では検出しにくい。でも「あるべき領域にあるべきものがあるか」を面で評価するなら、輪郭を基準に検査領域を切り出すアプローチが有効なはずだ。

この記事では、その経験をベースに画像検査の実装パターンを2種類紹介する。**外側パターン**（ワーク周囲のコーティング・フィルムの有無）と**内側パターン**（ワーク表面のラベルの有無と位置）だ。どちらも出発点は同じ――輪郭検出で「あるべき領域」を定義するところから始まる。

前回の記事（[搬送ライン上のワーク位置ずれをPythonで定量化する](https://zenn.dev/factory_dx_eng/articles/work-position-measurement)）ではテンプレートマッチングでワークの「位置」を見つけた。今回はその続きとして、「位置がわかった後に何を検査するか」を扱う。

**対象読者**：Python + OpenCV 初学者〜中級者、製造業 DX に興味のある方

---

## 共通技術：輪郭検出で「あるべき領域」を定義する

### なぜ輪郭検出から始めるのか

OK/NG を判定するには、まず「どの領域を見ればよいか」を決めなければならない。画像全体の輝度やヒストグラムをそのまま見ても、照明ムラやノイズに引っ張られる。ワークの輪郭を取得し、その輪郭を基準に ROI（検査領域）を定義する――この発想が各パターンの土台になる。

### 処理フロー

```
グレースケール → 二値化 → Opening（ノイズ除去）→ findContours → 面積フィルタ
```

ノイズ除去は 2 段構えにする。

**① Opening（前処理）**で二値画像の段階でノイズブロブを消す。Opening は「侵食→膨張」の組み合わせで、ノイズ由来の微小領域を除去しつつワーク形状は保つ処理だ。`findContours` に渡す輪郭数を事前に減らし、処理を安定させる。

**② `contourArea` フィルタ（後処理）**で、残った輪郭をサイズで選別する。照明ムラや背景の映り込みがある現場では、この 2 段構えが実用上ほぼ必須だ。

下の画像は Opening 前後の変化だ。左には微小なノイズブロブが散在しているが、右では円板（ワーク）の輪郭だけが残っている。

![Opening前（ノイズあり）](https://storage.googleapis.com/zenn-user-upload/5a70af254c87-20260405.png)
![Opening後（ノイズ除去）](https://storage.googleapis.com/zenn-user-upload/dd8bca036bc4-20260405.png)

### コード

```python
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# 設定パラメータ
# ============================================================
BINARY_THRESH    = 120    # 二値化の閾値
MIN_CONTOUR_AREA = 5000   # 面積フィルタ：これ以下の輪郭を除外(px²)
MORPH_KERNEL     = 9      # Opening カーネルサイズ(px)


def extract_contour(img_gray):
    """グレースケール画像からワークの輪郭を抽出する"""
    # 二値化（ワーク部分を白、背景を黒にする）
    _, binary = cv2.threshold(img_gray, BINARY_THRESH, 255, cv2.THRESH_BINARY_INV)

    # Opening：微小ノイズを除去してから findContours に渡す
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 輪郭検出
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積フィルタ：対象ワークの輪郭だけを残す
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not contours:
        raise RuntimeError("ワークの輪郭が検出できませんでした")

    return max(contours, key=cv2.contourArea)
```

`RETR_EXTERNAL` は最外周の輪郭のみを返すオプションだ。ワーク内側の模様や傷を拾わず、外形輪郭だけに絞れる。

---

## 外側パターン：コーティング・フィルムの有無を検査する

### 考え方

```
① ワーク輪郭を取得する
② 輪郭を外側に膨張させる（dilate）
③「膨張後 − 元輪郭」でリング状の ROI を作る
④ ROI 内の輝度ヒストグラムを正常品と比較する
```

外側リングには「コーティング材・フィルム・ガスケットなど周囲に付くべきもの」が写っている。正常品のヒストグラムと現品のヒストグラムを比べ、類似度が低ければ NG と判定する。

![外側パターン OK](https://storage.googleapis.com/zenn-user-upload/f6432f626c4f-20260405.png)
![外側パターン NG](https://storage.googleapis.com/zenn-user-upload/b3999779a8b7-20260405.png)

左が OK（コーティングリングあり）、右が NG（上方向に約 1/3 欠損）だ。緑ハイライトが「見ている領域」を示している。

### コード

```python
COATING_WIDTH  = 30     # 外側リングの幅(px)
HIST_THRESHOLD = 0.95   # ヒストグラム類似度の閾値（0〜1）


def make_outer_mask(img_shape, contour, width):
    """外側リングのマスクを生成する（膨張輪郭 − 元輪郭）"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2, width * 2))
    dilated = cv2.dilate(mask, kernel)

    # 外側リング = 膨張後 − 元の輪郭内部
    return cv2.subtract(dilated, mask)


def inspect_outer(img_gray, contour, ref_hist):
    """ヒストグラム類似度でコーティングの有無を判定する"""
    ring_mask = make_outer_mask(img_gray.shape, contour, COATING_WIDTH)

    hist = cv2.calcHist([img_gray], [0], ring_mask, [256], [0, 256])
    cv2.normalize(hist, hist)

    score = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
    return ("OK" if score >= HIST_THRESHOLD else "NG"), score
```

`compareHist` の戻り値は −1〜1 の相関係数で、1 に近いほど正常品のヒストグラムと一致する。基準ヒストグラムは事前に正常品から作成しておく。

```python
# 正常品から基準ヒストグラムを作成する
ref_img     = cv2.imread("samples/outer_ok.png", cv2.IMREAD_GRAYSCALE)
ref_contour = extract_contour(ref_img)
ref_mask    = make_outer_mask(ref_img.shape, ref_contour, COATING_WIDTH)
ref_hist    = cv2.calcHist([ref_img], [0], ref_mask, [256], [0, 256])
cv2.normalize(ref_hist, ref_hist)
```

---

## 内側パターン：同じ考え方は「内側」にも使える

外側リングを作るときは `dilate` で輪郭を外側に広げた。内側を見る場合は、輪郭をそのまま**塗りつぶす**だけでいい。ROI の作り方が変わるだけで、発想は共通だ。

```python
def make_inner_mask(img_shape, contour):
    """内側マスクを生成する（輪郭の内部を塗りつぶす）"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask
```

内側パターンの判定では「ラベルが存在するか」に加え、「正しい位置にあるか」も見る。ラベルの重心がワーク中心から離れすぎていれば位置ズレとして NG にする。

```python
LABEL_AREA_RATIO = 0.05   # ラベル面積比率の閾値（これ以下 → ラベルなし）
LABEL_OFFSET_MAX = 20     # ラベル重心の許容オフセット(px)


def inspect_inner(img_gray, contour):
    """ラベルの有無と位置（重心オフセット）を判定する"""
    inner_mask = make_inner_mask(img_gray.shape, contour)
    roi_area   = cv2.countNonZero(inner_mask)

    # ラベル（高輝度）領域を ROI 内で抽出
    _, label_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    label_region    = cv2.bitwise_and(label_binary, label_binary, mask=inner_mask)
    label_area      = cv2.countNonZero(label_region)
    ratio = label_area / roi_area if roi_area > 0 else 0

    if ratio < LABEL_AREA_RATIO:
        return "NG", ratio   # ラベルなし

    # ラベル重心とディスク中心のオフセットを計算
    disc_M   = cv2.moments(inner_mask)
    disc_cx  = disc_M["m10"] / disc_M["m00"]
    disc_cy  = disc_M["m01"] / disc_M["m00"]
    label_M  = cv2.moments(label_region)
    label_cx = label_M["m10"] / label_M["m00"]
    label_cy = label_M["m01"] / label_M["m00"]

    offset = np.sqrt((label_cx - disc_cx) ** 2 + (label_cy - disc_cy) ** 2)
    return ("NG" if offset > LABEL_OFFSET_MAX else "OK"), ratio
```

![内側パターン OK](https://storage.googleapis.com/zenn-user-upload/f127ee7cc950-20260405.png)
![内側パターン NG](https://storage.googleapis.com/zenn-user-upload/cf3876cfeed8-20260405.png)

左が OK（ラベルが中央付近）、右が NG（ラベルが右下方向に位置ズレ）だ。ratio は同じ 0.087 でも、重心オフセットが `LABEL_OFFSET_MAX` を超えるため NG と判定される。

---

## 現場に適用するときのポイント

### 二値化閾値の決め方

固定閾値（`cv2.threshold`）は画像全体に一定の閾値を適用するため、照度ムラがある現場では端と中央で判定がずれやすい。

**適応的二値化（`cv2.adaptiveThreshold`）** は局所領域ごとに閾値を計算するため、照度ムラに強い。

```python
binary = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=51,   # 局所領域のサイズ（奇数）
    C=5             # 閾値からの定数オフセット
)
```

ただし `blockSize` と `C` の調整が必要で、固定閾値より設定は複雑になる。まず固定閾値で試し、照度ムラで誤検出が出るようなら適応的二値化に切り替える、という順序が現実的だ。

### HSV範囲のキャリブレーション

色でラベルや塗装を判定する場合、HSV変換を使うと照明の明るさ変動に強くなる。H（色相）は輝度変化の影響を受けにくいからだ。

```python
hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
```

ただし H の範囲は照明の色温度や経時変化で少しずつずれる。定期的に正常品を撮影して範囲を確認する運用を組み込んでおくと安定する。

### NG判定基準の設定

`HIST_THRESHOLD = 0.95` のような閾値は、現場の正常品を複数撮影してスコア分布を確認してから決める。まず緩めの閾値で運用し、実績データを見ながら締めていくのが現実的だ。誤検出（OK品をNGと判定）が多すぎると現場の信頼を失うし、検知漏れが多いと検査の意味がなくなる。

### カメラ・照明の固定

カメラ位置や照明条件が変わると輪郭の形・ヒストグラムの分布が変わり、判定精度が落ちる。この点は[前回記事](https://zenn.dev/factory_dx_eng/articles/work-position-measurement)と同じ原則が適用される。

---

## 実装上の課題と対応アイデア

### 課題1：背景とワークのコントラストが低くて輪郭が取れない

透明・半透明ワークや背景と似た色のワークでよく起きる。

- **まず試すこと**：照明の角度を変えて**暗視野照明**（斜光）にする。エッジが浮き上がり、コントラストが大きく改善することが多い
- **それでもダメなら**：`cv2.normalize` で画像のコントラストを強調してから二値化する
- **根本対策**：照明の波長を変える（赤外・UV照明）。可視光では見えにくい材料の差が際立つことがある

### 課題2：品種ごとに色が違い、HSV範囲の管理が煩雑になる

品種数が増えるにつれて、各品種の HSV 範囲を個別に管理するのが負担になる。

- **まず試すこと**：品種コードをキーにした辞書で管理する。切替時はキーを変えるだけなのでコードの変更は最小限だ

```python
hsv_ranges = {
    "TYPE_A": ((10, 100, 100), (30, 255, 255)),
    "TYPE_B": ((100, 80, 80),  (130, 255, 255)),
}
lower, upper = hsv_ranges[current_type]
```

- **根本対策**：色の絶対値に依存しないヒストグラム比較に切り替える。`compareHist` のアプローチは内側パターンにも転用できる

### 課題3：輪郭が複数検出されて対象ワークを特定できない

複数のワークや背景オブジェクトが写り込むと、`findContours` が複数の輪郭を返す。

- **まず試すこと**：`MIN_CONTOUR_AREA` を調整して小さい輪郭を除外する
- **それでもダメなら**：外接矩形の位置（画像中央付近にあるか）や縦横比で対象を絞り込む。搬送ラインであればワークが毎回ほぼ同じ位置に来るはずなので、位置による絞り込みは有効だ
- **根本対策**：**バックライト照明**でワークをシルエットとして切り出す。背景との差が明確になり、輪郭検出が格段に安定する

---

## まとめ

| パターン | ROI の作り方 | 主な用途 |
|---|---|---|
| 外側パターン | `dilate` で膨張 → 元輪郭を引く | コーティング・フィルムの有無 |
| 内側パターン | `drawContours` で内部を塗りつぶす | ラベル・コーティングの有無と位置 |

ROI の作り方は違っても、「輪郭を基準に領域を定義し、その領域の特徴量で判定する」という発想は共通だ。検査項目が変わっても、このフレームワークはそのまま使い回せる。センサーで 1 点を見る代わりに、カメラで面を見る――そのアプローチの汎用性が、今回のコードに込めたものだ。

---

## GitHubリポジトリ

サンプルコード・サンプル画像一式を公開しています。

🔗 [github.com/factory-dx-eng/manufacturing-python](https://github.com/factory-dx-eng/manufacturing-python/tree/main/contour_inspection)

---
