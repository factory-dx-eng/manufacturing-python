"""
搬送ライン上のワーク位置ずれ定量化
テンプレートマッチングを使ってXY方向のずれをmmで出力する

使い方:
    python main.py

サンプル画像について:
    samples/ ディレクトリに以下を用意してください
    - samples/reference.png   : 基準画像（正常位置のワーク）
    - samples/test_01.png     : 検査画像（ずれあり）
    - samples/test_02.png     : 検査画像（ずれあり）
    ...

    サンプル画像がない場合は generate_samples.py で生成できます
"""

import cv2
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

# ============================================================
# 設定パラメータ
# ============================================================
PX_PER_MM = 10.0    # キャリブレーション係数 (px/mm)
                    # 求め方：既知寸法(mm) ÷ 画像上のピクセル数
                    # 例：100mmの治具が200pxで写っていれば 200/100=2.0

MARGIN = 80         # テンプレート切り出し時の余白(px)
                    # 検出可能な最大ずれ量 = MARGIN / PX_PER_MM (mm)

SCORE_THRESHOLD = 0.85  # マッチングスコアの閾値
                        # これを下回る場合は回転ずれ等の異常として警告

REFERENCE_PATH = "samples/reference.png"   # 基準画像のパス
TEST_DIR       = "samples"                  # 検査画像ディレクトリ
LOG_PATH       = "offset_log.csv"          # 計測ログの保存先


# ============================================================
# ブロック1: 画像の読み込み
# ============================================================
def load_image(path: str) -> np.ndarray:
    """グレースケールで画像を読み込む"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {path}")
    return img


def load_test_images(test_dir: str, reference_path: str) -> list:
    """検査画像を一括読み込みする（基準画像を除く）"""
    ref_name = Path(reference_path).name
    test_paths = sorted(
        p for p in Path(test_dir).glob("*.png")
        if p.name != ref_name
    )

    if not test_paths:
        raise FileNotFoundError(f"検査画像が見つかりません: {test_dir}")

    return [{"path": str(p), "image": load_image(str(p))} for p in test_paths]


# ============================================================
# ブロック2: テンプレートマッチングでずれ量(px)を取得
# ============================================================
def measure_offset_px(ref_img, test_img, margin=MARGIN):
    """
    テンプレートマッチングでずれ量(px)を返す

    Parameters
    ----------
    ref_img : 基準画像
    test_img : 検査画像
    margin  : テンプレート切り出し時の余白(px)
              検出可能な最大ずれ量 = margin / PX_PER_MM (mm)

    Returns
    -------
    dx_px, dy_px : XY方向のずれ量(px)  正値=右/下方向
    score        : マッチングスコア(0〜1)  低い場合は回転ずれ等を疑う
    """
    h, w = ref_img.shape

    # 基準画像の中央部をテンプレートとして切り出す
    # margin分の余白を残すことで、その範囲内のずれを検出できる
    tmpl = ref_img[margin:h - margin, margin:w - margin]

    # テンプレートマッチング（正規化相関係数法）
    # TM_CCOEFF_NORMED は照明の明るさ変動にやや強い
    result = cv2.matchTemplate(test_img, tmpl, cv2.TM_CCOEFF_NORMED)

    # スコアが最大の位置を取得
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # ずれなし(基準)のとき max_loc は (margin, margin) になる
    # そこからのずれがワークのXYオフセット
    dx_px = max_loc[0] - margin
    dy_px = max_loc[1] - margin

    return dx_px, dy_px, max_val


# ============================================================
# ブロック3: px→mm換算して結果を表示・保存
# ============================================================
def px_to_mm(dx_px, dy_px, px_per_mm=PX_PER_MM):
    """ピクセルをmmに換算する"""
    return dx_px / px_per_mm, dy_px / px_per_mm


def save_log(log_path, filename, dx_mm, dy_mm, score):
    """計測結果をCSVに追記する"""
    write_header = not Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "filename", "dx_mm", "dy_mm", "score"])
        writer.writerow([datetime.now().isoformat(), filename, dx_mm, dy_mm, score])


def run(reference_path=REFERENCE_PATH, test_dir=TEST_DIR):
    """メイン処理：基準画像と検査画像を読み込み、ずれ量を計測・表示する"""

    # 基準画像の読み込み
    ref_img = load_image(reference_path)
    print(f"基準画像: {reference_path}")

    # 検査画像の読み込み
    test_images = load_test_images(test_dir, reference_path)
    print(f"検査画像: {len(test_images)}件\n")

    print("=" * 60)
    print(f"{'ファイル名':<20} {'dX(mm)':>8} {'dY(mm)':>8}  {'スコア':>6}  {'判定':>4}")
    print("-" * 60)

    for case in test_images:
        dx_px, dy_px, score = measure_offset_px(ref_img, case["image"])
        dx_mm, dy_mm = px_to_mm(dx_px, dy_px)

        status = "OK" if score >= SCORE_THRESHOLD else "警告"
        filename = Path(case["path"]).name

        print(f"{filename:<20} {dx_mm:>8.2f} {dy_mm:>8.2f}  {score:>6.3f}  {status:>4}")

        if score < SCORE_THRESHOLD:
            print(f"  警告: スコア低下 - 回転ずれ・ワーク欠け等の可能性があります")

        save_log(LOG_PATH, filename, dx_mm, dy_mm, score)

    print("=" * 60)
    print(f"\nログ保存: {LOG_PATH}")


if __name__ == "__main__":
    run()
