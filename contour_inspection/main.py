import cv2
import numpy as np
from pathlib import Path

# ============================================================
# 設定パラメータ
# ============================================================
SAMPLES_DIR = Path("samples")

# 輪郭検出
BINARY_THRESH    = 120        # 二値化の閾値
MIN_CONTOUR_AREA = 5000       # 面積フィルタ：これ以下の輪郭を除外(px²)
MORPH_KERNEL     = 9          # Openingカーネルサイズ(px)

# 外側パターン
COATING_WIDTH    = 30         # 外側リングの幅(px)
HIST_THRESHOLD   = 0.95       # ヒストグラム類似度の判定閾値（0〜1、高いほど厳しい）

# 内側パターン
LABEL_AREA_RATIO  = 0.05      # ラベル面積比率の閾値（これ以下 → ラベルなしとしてNG）
LABEL_OFFSET_MAX  = 20        # ラベル重心の許容オフセット(px)（これ以上 → 位置ズレとしてNG）


# ============================================================
# ブロック1：共通処理 ― 輪郭検出で「あるべき領域」を定義する
# ============================================================

def extract_contour(img_gray):
    """
    グレースケール画像からワークの輪郭を抽出する

    処理フロー:
        二値化 → Opening（小領域除去）→ findContours → 面積フィルタ
    """
    # 二値化（ワーク部分を白、背景を黒にする）
    _, binary = cv2.threshold(img_gray, BINARY_THRESH, 255, cv2.THRESH_BINARY_INV)

    # Opening：微小ノイズを除去してからfindContoursに渡す
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 輪郭検出
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積フィルタ：対象ワークの輪郭だけを残す
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]

    if not contours:
        raise RuntimeError("ワークの輪郭が検出できませんでした")

    # 最大面積の輪郭を対象ワークとして返す
    return max(contours, key=cv2.contourArea)


def draw_contour(img_gray, contour):
    """輪郭を可視化した画像を返す"""
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
    return vis


# ============================================================
# ブロック2：外側パターン ― コーティング・フィルムの有無を検査する
# ============================================================

def make_outer_mask(img_shape, contour, width):
    """外側リングのマスクを生成する（膨張輪郭 − 元輪郭）"""
    h, w = img_shape[:2]
    mask_inner = np.zeros((h, w), dtype=np.uint8)
    mask_outer = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(mask_inner, [contour], -1, 255, -1)

    # 輪郭を外側に膨張させてリングの外縁を作る
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2, width * 2))
    dilated = cv2.dilate(mask_inner, kernel)
    cv2.drawContours(mask_outer, [contour], -1, 255, -1)

    # 外側リング = 膨張後 − 元の輪郭内部
    return cv2.subtract(dilated, mask_outer)


def inspect_outer(img_gray, contour, ref_hist):
    """
    外側パターン検査：ヒストグラム類似度でコーティングの有無を判定する

    Parameters
    ----------
    ref_hist : 正常品（OK）の外側リング領域ヒストグラム
    """
    ring_mask = make_outer_mask(img_gray.shape, contour, COATING_WIDTH)

    hist = cv2.calcHist([img_gray], [0], ring_mask, [256], [0, 256])
    cv2.normalize(hist, hist)

    score = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)
    result = "OK" if score >= HIST_THRESHOLD else "NG"
    return result, score, ring_mask


def draw_outer_result(img_gray, contour, ring_mask, result, score):
    """外側パターンの判定結果を可視化する"""
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color = (0, 200, 0) if result == "OK" else (0, 0, 220)

    # リング領域をハイライト
    highlight = np.zeros_like(vis)
    highlight[ring_mask > 0] = color
    vis = cv2.addWeighted(vis, 1.0, highlight, 0.4, 0)

    cv2.drawContours(vis, [contour], -1, color, 2)
    cv2.putText(vis, f"{result}  score={score:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis


# ============================================================
# ブロック3：内側パターン ― ラベル・コーティングの有無を検査する
# ============================================================

def make_inner_mask(img_shape, contour):
    """内側マスクを生成する（輪郭の内部を塗りつぶす）"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def inspect_inner(img_gray, contour):
    """
    内側パターン検査：ラベルの有無と位置（重心オフセット）を判定する

    ① ラベル面積比率 < LABEL_AREA_RATIO → ラベルなし（NG）
    ② ラベル重心がディスク中心から LABEL_OFFSET_MAX 以上離れている → 位置ズレ（NG）
    """
    inner_mask = make_inner_mask(img_gray.shape, contour)
    roi_area = cv2.countNonZero(inner_mask)

    # ラベル（高輝度）領域を抽出
    _, label_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    label_region = cv2.bitwise_and(label_binary, label_binary, mask=inner_mask)
    label_area = cv2.countNonZero(label_region)

    ratio = label_area / roi_area if roi_area > 0 else 0

    if ratio < LABEL_AREA_RATIO:
        return "NG", ratio, inner_mask   # ラベルなし

    # ラベル重心とディスク中心のオフセットを計算
    disc_M  = cv2.moments(inner_mask)
    disc_cx = disc_M["m10"] / disc_M["m00"]
    disc_cy = disc_M["m01"] / disc_M["m00"]

    label_M  = cv2.moments(label_region)
    label_cx = label_M["m10"] / label_M["m00"]
    label_cy = label_M["m01"] / label_M["m00"]

    offset = np.sqrt((label_cx - disc_cx) ** 2 + (label_cy - disc_cy) ** 2)
    result = "NG" if offset > LABEL_OFFSET_MAX else "OK"
    return result, ratio, inner_mask


def draw_inner_result(img_gray, contour, inner_mask, result, ratio):
    """内側パターンの判定結果を可視化する"""
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color = (0, 200, 0) if result == "OK" else (0, 0, 220)

    # 内側領域をハイライト
    highlight = np.zeros_like(vis)
    highlight[inner_mask > 0] = color
    vis = cv2.addWeighted(vis, 1.0, highlight, 0.3, 0)

    cv2.drawContours(vis, [contour], -1, color, 2)
    cv2.putText(vis, f"{result}  ratio={ratio:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis


# ============================================================
# 実行
# ============================================================

def run():
    # --- 外側パターン ---
    print("=== 外側パターン検査 ===")

    # OK画像から基準ヒストグラムを作成
    ref_outer = cv2.imread(str(SAMPLES_DIR / "outer_ok.png"), cv2.IMREAD_GRAYSCALE)
    ref_contour = extract_contour(ref_outer)
    ref_ring_mask = make_outer_mask(ref_outer.shape, ref_contour, COATING_WIDTH)
    ref_hist = cv2.calcHist([ref_outer], [0], ref_ring_mask, [256], [0, 256])
    cv2.normalize(ref_hist, ref_hist)

    for fname in ["outer_ok.png", "outer_ng.png"]:
        img = cv2.imread(str(SAMPLES_DIR / fname), cv2.IMREAD_GRAYSCALE)
        contour = extract_contour(img)
        result, score, ring_mask = inspect_outer(img, contour, ref_hist)
        vis = draw_outer_result(img, contour, ring_mask, result, score)
        out_path = SAMPLES_DIR / f"result_{fname}"
        cv2.imwrite(str(out_path), vis)
        print(f"  {fname:<20} → {result}  (score={score:.3f})")

    print()

    # --- 内側パターン ---
    print("=== 内側パターン検査 ===")

    for fname in ["inner_ok.png", "inner_ng.png"]:
        img = cv2.imread(str(SAMPLES_DIR / fname), cv2.IMREAD_GRAYSCALE)
        contour = extract_contour(img)
        result, ratio, inner_mask = inspect_inner(img, contour)
        vis = draw_inner_result(img, contour, inner_mask, result, ratio)
        out_path = SAMPLES_DIR / f"result_{fname}"
        cv2.imwrite(str(out_path), vis)
        print(f"  {fname:<20} → {result}  (ratio={ratio:.3f})")


if __name__ == "__main__":
    run()
