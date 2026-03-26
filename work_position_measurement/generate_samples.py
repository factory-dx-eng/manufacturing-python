"""
サンプル画像生成スクリプト
実画像がない環境での動作確認・サンプル追加用

使い方:
    python generate_samples.py

実行すると samples/ ディレクトリに以下を生成します
    - reference.png                  : 基準画像（ずれなし）
    - test_dx+3.0_dy+0.0.png         : 検査画像（ファイル名にずれ量を記載）
    - ...

サンプルを追加したい場合:
    OFFSET_CASES にずれ量 (dx_mm, dy_mm) を追加して実行してください
"""

import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = (400, 400)
PX_PER_MM = 10.0

# 生成する検査画像のずれ量リスト (dx_mm, dy_mm)
# ここに追加することでサンプルを増やせます
OFFSET_CASES = [
    ( 3.0,  0.0),
    ( 0.0, -2.0),
    ( 4.0,  3.0),
    (-3.0,  2.5),
]


def generate_work_image(img_size, offset_px=(0, 0)):
    """不定形ワークの疑似画像を生成する"""
    h, w = img_size
    img = np.ones((h, w), dtype=np.uint8) * 200  # 背景：明るいグレー

    cx = w // 2 + offset_px[0]
    cy = h // 2 + offset_px[1]

    # 不定形ワーク：多角形＋突起で表現
    pts_main = np.array([
        [cx - 60, cy - 30],
        [cx - 20, cy - 70],
        [cx + 40, cy - 60],
        [cx + 70, cy - 10],
        [cx + 50, cy + 50],
        [cx - 30, cy + 60],
        [cx - 70, cy + 20],
    ], dtype=np.int32)

    pts_notch = np.array([
        [cx + 40, cy - 60],
        [cx + 80, cy - 80],
        [cx + 90, cy - 40],
        [cx + 70, cy - 10],
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts_main], color=80)
    cv2.fillPoly(img, [pts_notch], color=80)
    cv2.circle(img, (cx - 20, cy + 10), 12, 200, -1)
    cv2.circle(img, (cx + 20, cy - 20),  8, 200, -1)

    return img


def offset_to_filename(dx_mm, dy_mm):
    """ずれ量をファイル名に変換する  例: test_dx+3.0_dy-2.0.png"""
    return f"test_dx{dx_mm:+.1f}_dy{dy_mm:+.1f}.png"


def main():
    output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(exist_ok=True)

    # 基準画像
    ref_img = generate_work_image(IMG_SIZE, offset_px=(0, 0))
    cv2.imwrite(str(output_dir / "reference.png"), ref_img)
    print("生成: samples/reference.png  (ずれなし)")

    # 検査画像
    for dx_mm, dy_mm in OFFSET_CASES:
        dx_px = int(round(dx_mm * PX_PER_MM))
        dy_px = int(round(dy_mm * PX_PER_MM))
        img = generate_work_image(IMG_SIZE, offset_px=(dx_px, dy_px))
        filename = offset_to_filename(dx_mm, dy_mm)
        cv2.imwrite(str(output_dir / filename), img)
        print(f"生成: samples/{filename}")

    print(f"\nsamples/ に {1 + len(OFFSET_CASES)} 枚の画像を生成しました")
    print("次のコマンドで計測を実行できます: python main.py")


if __name__ == "__main__":
    main()
