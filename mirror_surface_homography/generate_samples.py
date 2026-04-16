"""
サンプル画像生成スクリプト
実画像がない環境での動作確認用

使い方:
    python generate_samples.py

実行すると samples/ ディレクトリに以下を生成します
    - frontal.png  : 正面から見たワーク（グラウンドトゥルース確認用）
    - angled.png   : 斜め撮像を再現した画像（main.py の入力）

曇りパッチを変えたい場合:
    FOG_PATCHES の内容を変更して実行してください
"""

import cv2
import numpy as np
from pathlib import Path

# ---- パラメータ --------------------------------------------------------
IMG_W, IMG_H  = 640, 480    # 画像サイズ [px]
WORK_RADIUS   = 190         # ワーク（円形）の半径 [px]
WORK_CENTER   = (320, 240)  # ワーク中心座標 [px]

# 曇りパッチ（複数指定可）: (中心x, 中心y, 半径)
FOG_PATCHES = [
    (260, 200, 55),
    (350, 270, 40),
]

# 斜め撮像の再現：正面画像の四隅が斜め画像のどこに対応するかを定義
# （左辺を右側に寄せることで、カメラが右斜め上から撮った状態を再現）
SRC_PTS = np.float32([
    [0,     0    ],
    [IMG_W, 0    ],
    [IMG_W, IMG_H],
    [0,     IMG_H],
])
DST_PTS = np.float32([
    [180,  50   ],   # 左上 → 右・下に寄る（遠ざかる側）
    [IMG_W - 30, 20  ],   # 右上 → ほぼそのまま（近い側）
    [IMG_W - 30, IMG_H - 20],   # 右下
    [180,  IMG_H - 50],   # 左下 → 右に寄る
])
# -----------------------------------------------------------------------


def generate_mirror_workpiece() -> np.ndarray:
    """鏡面ワーク（Siターゲット）のダミー画像を生成する

    暗幕ブース＋拡散光の環境を想定:
    - 鏡面（正常部）: 光が鏡面反射してカメラに入らず暗く写る（30〜75）
    - 曇り部: 表面が粗くなり光を散乱→カメラに届いて明るく写る（140）
    """
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)  # 黒背景（暗幕ブース）

    cx, cy = WORK_CENTER

    Y, X = np.ogrid[:IMG_H, :IMG_W]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
    on_work = dist <= WORK_RADIUS

    # 鏡面部: 外周ほど僅かに明るい緩やかなグラデーション（30〜75）
    # 中心は暗く、外周に向かって少し明るくなる（リング照明の反射を模倣）
    grad = dist / WORK_RADIUS  # 0（中心）→ 1（外周）
    brightness = (grad * 45 + 30).astype(np.uint8)  # 30〜75 の範囲

    for c in range(3):
        ch = img[:, :, c]
        ch[on_work] = brightness[on_work]
        img[:, :, c] = ch

    # 曇りパッチ: 均一な明るさ（140）で鏡面部より明確に明るい
    for fx, fy, fr in FOG_PATCHES:
        fog_dist = np.sqrt((X - fx) ** 2 + (Y - fy) ** 2)
        fog_area = (fog_dist <= fr) & on_work
        img[fog_area] = [140, 140, 140]

    # ワーク外周のエッジ（治具の縁として対応点の目印になる）
    cv2.circle(img, (cx, cy), WORK_RADIUS, (200, 200, 200), 2)

    return img


def compute_ellipse_extremes() -> np.ndarray:
    """斜め画像上の楕円4点（上・右・下・左）を計算して返す

    正面画像でのワーク円の4端点を透視変換して求める。
    main.py の ANGLED_CORNERS に使用する値でもある。
    """
    M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
    circle_pts = np.float32([
        [WORK_CENTER[0],               WORK_CENTER[1] - WORK_RADIUS],  # 上
        [WORK_CENTER[0] + WORK_RADIUS, WORK_CENTER[1]              ],  # 右
        [WORK_CENTER[0],               WORK_CENTER[1] + WORK_RADIUS],  # 下
        [WORK_CENTER[0] - WORK_RADIUS, WORK_CENTER[1]              ],  # 左
    ])
    return cv2.perspectiveTransform(
        circle_pts.reshape(1, -1, 2), M
    ).reshape(-1, 2)


def generate_corner_selection(angled: np.ndarray,
                               ellipse_pts: np.ndarray) -> np.ndarray:
    """斜め画像に楕円4点のマーカーを描画する

    GUIツールで4点を指定するイメージを再現した説明用画像。
    点の順序: 上・右・下・左
    """
    img = angled.copy()
    pts = ellipse_pts.astype(np.int32)
    labels = ["1", "2", "3", "4"]  # 1=上, 2=右, 3=下, 4=左

    for (x, y), label in zip(pts, labels):
        # 赤い円
        cv2.circle(img, (x, y), 12, (0, 0, 220), -1)
        cv2.circle(img, (x, y), 12, (255, 255, 255), 2)
        # ラベル
        cv2.putText(img, label, (x - 5, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return img


def main():
    output_dir = Path(__file__).parent / "samples"
    output_dir.mkdir(exist_ok=True)

    # 正面画像（グラウンドトゥルース）
    frontal = generate_mirror_workpiece()
    cv2.imwrite(str(output_dir / "frontal.png"), frontal)
    print("生成: samples/frontal.png  (正面・グラウンドトゥルース確認用)")

    # 射影変換で斜め撮像を再現
    M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
    angled = cv2.warpPerspective(frontal, M, (IMG_W, IMG_H))
    cv2.imwrite(str(output_dir / "angled.png"), angled)
    print("生成: samples/angled.png   (斜め撮像を再現・main.py の入力)")

    # 対応点マーカー付き画像（記事用説明画像）
    ellipse_pts = compute_ellipse_extremes()
    corner_sel = generate_corner_selection(angled, ellipse_pts)
    cv2.imwrite(str(output_dir / "corner_selection.png"), corner_sel)
    print("生成: samples/corner_selection.png (対応点指定のイメージ画像)")

    print(f"\nsamples/ に 3 枚の画像を生成しました")
    print("次のコマンドで補正・計測を実行できます: python main.py")
    print(f"\n[参考] main.py の ANGLED_CORNERS に使う楕円4点の座標:")
    print(ellipse_pts)


if __name__ == "__main__":
    main()
