import cv2
import numpy as np
from pathlib import Path

# ============================================================
# 設定パラメータ
# ============================================================
OUTPUT_DIR    = Path("samples")
IMG_SIZE      = (400, 400)
CENTER        = (200, 200)
WORK_RADIUS   = 110          # ワーク半径(px)
COATING_WIDTH = 20           # 外側コーティング幅(px)
LABEL_SIZE    = (90, 55)     # 内側ラベルサイズ (幅, 高さ)(px)

NOISE_COUNT = 40             # ノイズ点の数（セクション2用）
NOISE_SEED  = 42
NOISE_MAX_R = 3              # ノイズ半径の上限(px)。Openingカーネルより小さくする


# ============================================================
# パーツ生成
# ============================================================

def make_background(seed=0):
    """検査台（明るいグレー＋テクスチャノイズ）"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 4, IMG_SIZE)
    return np.clip(198 + noise, 170, 230).astype(np.float64)


def make_disc(seed=1):
    """
    金属円板（放射グラデーション＋旋盤加工テクスチャ＋表面ノイズ）

    Returns
    -------
    disc     : float64 array, ワーク外は 0
    work_mask: bool array
    """
    h, w = IMG_SIZE
    cx, cy = CENTER
    Y, X = np.mgrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    work_mask = dist <= WORK_RADIUS

    rng = np.random.default_rng(seed)

    # ベース輝度（暗いスチールグレー）
    disc = np.full((h, w), 90.0)

    # 外縁に向かってわずかに暗くなる放射グラデーション
    disc -= 12 * np.clip(dist / WORK_RADIUS, 0, 1)

    # 同心円状の輝度変化（旋盤加工テクスチャ）
    disc += 4 * np.sin(25 * np.pi * dist / WORK_RADIUS)

    # 表面ノイズ
    disc += rng.normal(0, 4, (h, w))

    disc = np.where(work_mask, disc, 0.0)
    return disc, work_mask


def make_base_image():
    """検査台 + 金属円板の合成グレースケール画像"""
    bg = make_background(seed=0)
    disc, work_mask = make_disc(seed=1)
    img = np.where(work_mask, disc, bg)
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# サンプル画像生成
# ============================================================

def make_outer_ok():
    """外側パターン正常品：ゴムパッキン/保護フィルムリングあり"""
    h, w = IMG_SIZE
    cx, cy = CENTER
    Y, X = np.mgrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    bg = make_background(seed=0)
    disc, work_mask = make_disc(seed=1)

    # コーティングリング（背景より暗く、ワークより明るい中間色）
    ring_mask = (dist > WORK_RADIUS) & (dist <= WORK_RADIUS + COATING_WIDTH)
    rng = np.random.default_rng(2)
    coating = np.clip(155 + rng.normal(0, 3, (h, w)), 140, 175)

    img = np.where(work_mask, disc, bg)
    img = np.where(ring_mask, coating, img)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_outer_ng():
    """外側パターン異常品：コーティングが一部欠損（ガスケット浮き/フィルム未着）"""
    h, w = IMG_SIZE
    cx, cy = CENTER
    Y, X = np.mgrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    bg = make_background(seed=0)
    disc, work_mask = make_disc(seed=1)

    # コーティングリング（OK と同素材）
    ring_mask = (dist > WORK_RADIUS) & (dist <= WORK_RADIUS + COATING_WIDTH)
    rng = np.random.default_rng(2)
    coating = np.clip(155 + rng.normal(0, 3, (h, w)), 140, 175)

    # 欠損エリア：右上〜左上方向 (20°〜140°) を除外（全周の約1/3）
    angle = np.degrees(np.arctan2(Y - cy, X - cx))  # -180〜180
    gap_mask = (angle >= 20) & (angle <= 140)
    ring_mask = ring_mask & ~gap_mask

    img = np.where(work_mask, disc, bg)
    img = np.where(ring_mask, coating, img)
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_label(img, label_cx, label_cy):
    """製品ラベルを指定中心座標に描画する（make_inner_ok/ng で共用）"""
    lw, lh = LABEL_SIZE
    x1, y1 = label_cx - lw // 2, label_cy - lh // 2
    x2, y2 = label_cx + lw // 2, label_cy + lh // 2

    cv2.rectangle(img, (x1, y1), (x2, y2), 238, -1)   # 白背景
    cv2.rectangle(img, (x1, y1), (x2, y2), 165, 1)    # 枠線

    lines = [
        (y1 + 9,  lw - 16, 5),       # タイトル行（太め）
        (y1 + 21, lw - 26, 3),       # 本文行
        (y1 + 30, lw - 20, 3),
        (y1 + 39, lw - 28, 3),
        (y1 + 48, lw // 2 - 8, 3),   # 短い行（ロット番号など）
    ]
    for ly, ll, thickness in lines:
        xs = label_cx - ll // 2
        xe = label_cx + ll // 2
        cv2.rectangle(img, (xs, ly), (xe, ly + thickness), 75, -1)


def make_inner_ok():
    """内側パターン正常品：ラベルが中央に貼られている"""
    img = make_base_image().astype(np.float64)
    cx, cy = CENTER
    _draw_label(img, cx, cy)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_inner_ng():
    """内側パターン異常品：ラベル貼り位置ズレ（右下方向にオフセット）"""
    img = make_base_image().astype(np.float64)
    cx, cy = CENTER
    _draw_label(img, cx + 25, cy + 20)    # 中心から右下方向にオフセット
    return np.clip(img, 0, 255).astype(np.uint8)


def make_binary_before_opening():
    """セクション2用：Opening前の二値画像（ノイズあり）"""
    img = make_base_image().copy()
    rng = np.random.default_rng(NOISE_SEED)
    for _ in range(NOISE_COUNT):
        x = int(rng.integers(10, IMG_SIZE[1] - 10))
        y = int(rng.integers(10, IMG_SIZE[0] - 10))
        r = int(rng.integers(1, NOISE_MAX_R + 1))
        cv2.circle(img, (x, y), r, 90, -1)
    _, binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    return binary


def make_binary_after_opening():
    """セクション2用：Opening後の二値画像（ノイズ除去済み）"""
    binary = make_binary_before_opening()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


# ============================================================
# 保存・実行
# ============================================================

def save(name, img):
    path = OUTPUT_DIR / name
    cv2.imwrite(str(path), img)
    print(f"保存: {path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    save("outer_ok.png",              make_outer_ok())
    save("outer_ng.png",              make_outer_ng())
    save("inner_ok.png",              make_inner_ok())
    save("inner_ng.png",              make_inner_ng())
    save("binary_before_opening.png", make_binary_before_opening())
    save("binary_after_opening.png",  make_binary_after_opening())

    print("\n完了")


if __name__ == "__main__":
    main()
