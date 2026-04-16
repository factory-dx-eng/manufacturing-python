"""
射影変換で斜め撮像画像を正面補正し、曇り面積を定量化するスクリプト

使い方:
    python main.py

入力: samples/angled.png（斜め撮像画像）
出力:
    - results/corrected.png  : 射影変換で正面補正した画像
    - results/binary.png     : 曇り領域の二値化画像
    - results/log.csv        : 曇り面積の計測ログ
"""

import csv
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# ---- パラメータ --------------------------------------------------------
IMG_W, IMG_H = 640, 480

# 斜め撮像画像上の対応点（楕円の上・右・下・左）
# ※ 実運用では GUI で楕円外周の4点をクリックして取得し、JSON で保存して固定する
ANGLED_CORNERS = np.float32([
    [379.3, 78.6 ],   # 上
    [512.0, 240.0],   # 右
    [379.3, 401.4],   # 下
    [257.6, 240.0],   # 左
])

# 補正後の正面画像上での対応点（円の上・右・下・左）
# WORK_CENTER=(320,240), WORK_RADIUS=190 に対応
FRONTAL_CORNERS = np.float32([
    [320, 50 ],   # 上
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
# 鏡面部（30〜75）と曇り部（140）の間に設定する
FOG_THRESHOLD = 110
# -----------------------------------------------------------------------


def correct_perspective(img: np.ndarray) -> np.ndarray:
    """射影変換で斜め画像を正面補正する"""
    M = cv2.getPerspectiveTransform(ANGLED_CORNERS, FRONTAL_CORNERS)
    return cv2.warpPerspective(img, M, (IMG_W, IMG_H))


def measure_fog_area(corrected: np.ndarray) -> tuple[np.ndarray, int, float]:
    """補正画像から曇り領域を検出し面積を算出する"""
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    # ワーク領域（円形）だけを検査対象にする
    work_mask = np.zeros_like(gray)
    cv2.circle(work_mask, WORK_CENTER, WORK_RADIUS, 255, -1)

    # 固定閾値で曇り領域を抽出する
    # 鏡面部（30〜75）と曇り部（140）の輝度差が大きいため単純な閾値で分離できる
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
    if angled is None:
        raise FileNotFoundError(f"画像が見つかりません: {input_path}")

    corrected = correct_perspective(angled)
    cv2.imwrite(str(results_dir / "corrected.png"), corrected)

    # ② 曇り面積の算出
    binary, fog_px, fog_mm2 = measure_fog_area(corrected)
    cv2.imwrite(str(results_dir / "binary.png"), binary)

    save_log(results_dir / "log.csv", fog_px, fog_mm2)

    print(f"曇り面積: {fog_px} px²  →  {fog_mm2:.1f} mm²")
    print("結果を results/ に保存しました")
    print("  - results/corrected.png  （補正後画像）")
    print("  - results/binary.png     （二値化画像）")
    print("  - results/log.csv        （計測ログ）")


if __name__ == "__main__":
    main()
