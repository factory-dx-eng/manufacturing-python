---
title: "製造ラインの停止原因を数値で特定する：チョコ停先頭ワーク判定をPandasで実装する"
emoji: "⏱️"
type: "tech"
topics: ["python", "pandas", "製造業", "DX", "データ分析"]
published: true
---

## 「プレスが多い気がします」——誰も数値で答えられない

「プレスが多い気がします」——そう答えた担当者も、自信なさげだった。

日次の生産実績を確認すると、計画に対して5〜10%のロスがある。不良品が多いわけでも、段取り替えが長かったわけでもない。記録を辿ると「その他ロス」の積み上げで説明がつく。いわゆるチョコ停だ。

「どこで止まることが多いですか？」と問うたびに、「プレスが多い気がします」「最近は研磨が怪しい」という答えが返ってくる。感覚で答えるしかない。なぜなら、チョコ停は「ちょっと止まっただけ」と流されて記録に残らないからだ。

一方で、PLCのログはDBに入っている。各設備の入口でワークを検知するたびにタイムスタンプが記録されている。ログは存在するが、誰も開いていない。

**このログから「どの設備で、どのワークを先頭にチョコ停が起きたか」を自動で判定できれば、感覚頼りの議論が変わる。** PLCのログが眠っている現場なら、今日から試せる。

---

## 単純に集計すると、原因でない設備が悪者になる

チョコ停の分析で最初にはまる落とし穴がある。

連続ラインでは、1か所で止まると後続ワークが詰まって下流設備も遅れて見える。たとえばプレスが30秒止まったとする。その間に研磨・検査・梱包の前にもワークが溜まり、プレスが再稼働した後もしばらくの間、下流の設備は「前のワークがまだ処理中」という状態で待ちが発生する。

タクト遅れをそのまま集計すると、「渋滞を受けただけ」の下流設備が回数の多い設備としてカウントされる。「プレスより研磨の方がチョコ停回数が多い」という集計結果が出ても、実態は「プレスで止まった渋滞の波が研磨まで届いた」だけかもしれない。

この問題を解決するのが**先頭ワーク判定**だ。

---

## アイデアの核心：先頭ワークを見つけて原因設備に帰属する

### データ構造

PLCログをDBから取得すると、次のような wide format のテーブルになる。1行が1ワーク、各列が設備ごとの入口通過時刻だ。

```
work_id | furnace_in_time     | press_in_time       | polish_in_time      | ...
W-014   | 2024-03-15 08:06:30 | 2024-03-15 08:15:00 | 2024-03-15 08:22:30 | ...
W-015   | 2024-03-15 08:07:00 | 2024-03-15 08:17:30 | 2024-03-15 08:24:00 | ...
W-016   | 2024-03-15 08:07:30 | 2024-03-15 08:18:30 | 2024-03-15 08:25:30 | ...
```

W-015 の `press_in_time` を見ると、W-014 から2分30秒あいている。標準タクトが30秒なら、90秒の遅れだ。

### タクトと先頭判定のロジック

```
① 各設備のタクト = 自分が入った時刻 − 前のワークが入った時刻

② タクトが「標準タクト × 閾値（例：1.5倍）」を超えたら「タクト遅れ」

③ タクト遅れ かつ 直前のワークが正常 → 「先頭ワーク」（= その設備に帰属）
   タクト遅れ かつ 直前のワークもすでに遅れていた → 「後続ワーク」（= 渋滞の波及、除外）

④ 同じ work_id が上流設備ですでに先頭判定済み → 「下流への波」（= 除外）
```

判定のフローを整理すると次のようになる。

```
[タクト計算] → [閾値超え?] → YES → [前ワークも遅れ?] → YES → 後続扱い（スキップ）
                                                       → NO  → 先頭ワーク（帰属）
                            → NO  → 正常
```

④ の処理がポイントだ。プレスで止まった渋滞が研磨まで伝播すると、研磨でも W-015 がタクト遅れとして検出される。しかし W-015 はすでにプレスで先頭判定されているので、研磨ではカウントしない。原因はプレスにある、と正しく帰属できる。

---

## 実装

コードは3つのブロックで構成する。サンプルデータは `generate_samples.py` でリポジトリに同梱しているので、手元で動かして確認できる。

### ブロック1：データ読み込みとタクト計算

```python
import pandas as pd

# ============================================================
# 設定パラメータ
# ============================================================
CSV_PATH       = 'samples/choco_stop_log.csv'
MACHINES       = ['furnace', 'press', 'polish', 'inspect', 'pack']
TACT_THRESHOLD = 1.5   # 標準タクトの何倍以上をチョコ停と判定するか
# ============================================================


def load_data(path):
    df = pd.read_csv(path)
    for m in MACHINES:
        df[f'{m}_in_time'] = pd.to_datetime(df[f'{m}_in_time'])
    return df


def calc_tact(df):
    for m in MACHINES:
        df[f'{m}_tact'] = df[f'{m}_in_time'].diff().dt.total_seconds()
    return df


df = load_data(CSV_PATH)
df = calc_tact(df)
```

`.diff()` で前行との差分を一括計算できる。`dt.total_seconds()` で秒数に変換している。

実際の現場では `pd.read_csv` を `pd.read_sql` に差し替えるだけで DB から直接読み込める。その後の処理は変わらない。

### ブロック2：先頭ワーク判定

```python
def detect_leaders(df):
    # 標準タクトをログの中央値で推定する（平均は外れ値に引っ張られるため不向き）
    standard_tacts = {m: df[f'{m}_tact'].median() for m in MACHINES}

    events = []
    for m in MACHINES:
        tact_col  = f'{m}_tact'
        threshold = standard_tacts[m] * TACT_THRESHOLD

        is_slow   = df[tact_col] > threshold
        # 前ワークも遅れていた場合は渋滞の後続なので除外する
        is_leader = is_slow & ~is_slow.shift(1, fill_value=False)

        for idx in df[is_leader].index:
            # 先頭から連続するタクト遅れを積算して停止時間を推定する
            total_delay = 0.0
            for j in range(idx, len(df)):
                t = df.loc[j, tact_col]
                if t > threshold:
                    total_delay += t - standard_tacts[m]
                else:
                    break

            events.append({
                'machine':      m,
                'work_id':      df.loc[idx, 'work_id'],
                'work_index':   idx,
                'entry_time':   df.loc[idx, f'{m}_in_time'],
                'duration_sec': total_delay,
            })

    leaders = pd.DataFrame(events)

    # 同一 work_id が上流設備で先頭判定済みなら下流の検出は除外する
    upstream_works = set()
    filtered = []
    for m in MACHINES:
        for _, ev in leaders[leaders['machine'] == m].iterrows():
            if ev['work_id'] not in upstream_works:
                filtered.append(ev)
                upstream_works.add(ev['work_id'])

    return pd.DataFrame(filtered), standard_tacts


leaders_df, standard_tacts = detect_leaders(df)
print(leaders_df[['machine', 'work_id', 'entry_time', 'duration_sec']])
```

`upstream_works` は「すでに上流で先頭判定された `work_id` の集合」だ。設備を上流から順に処理し、同じ `work_id` が登場したらスキップすることで、渋滞の波及先を結果から取り除いている。

出力例：

```
machine work_id          entry_time  duration_sec
  press   W-015 2024-03-15 08:17:30          90.0
  press   W-035 2024-03-15 08:38:30          60.0
 polish   W-026 2024-03-15 08:41:45          75.0
inspect   W-042 2024-03-15 09:09:15         120.0
```

プレス・研磨・検査の3設備でチョコ停が検出された。プレスで止まった渋滞が研磨や検査に波及した分は、正しく除外されている。

実際にこのロジックを動かしたとき、単純集計でワースト1位だった研磨が、先頭ワーク帰属後には3位に下がった。原因はプレスだった。現場の「研磨が怪しい」という感覚は、渋滞の影響を見ていたに過ぎなかった。

### ブロック3：可視化

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

MACHINE_LABELS = ['① 炉', '② プレス', '③ 研磨', '④ 検査', '⑤ 梱包']
COLORS         = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']

color_map = dict(zip(MACHINES, COLORS))
label_map = dict(zip(MACHINES, MACHINE_LABELS))
base_time = df['furnace_in_time'].min()

def to_sec(t):
    return (t - base_time).total_seconds()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# ── ガントチャート：チョコ停の発生タイミングと継続時間 ──
for y, m in enumerate(MACHINES):
    # 全ワークの通過リズムを細いバーで表示する
    for _, row in df.iterrows():
        ax1.barh(y, 20, left=to_sec(row[f'{m}_in_time']),
                 height=0.25, color=color_map[m], alpha=0.2)
    # チョコ停（先頭ワーク）を太いバーで強調する
    for _, ev in leaders_df[leaders_df['machine'] == m].iterrows():
        dur = max(ev['duration_sec'], 40)
        ax1.barh(y, dur, left=to_sec(ev['entry_time']),
                 height=0.6, color=color_map[m], alpha=0.9)
        ax1.text(to_sec(ev['entry_time']) + 5, y,
                 ev['work_id'], va='center', fontsize=7.5,
                 color='white', fontweight='bold')

ax1.set_yticks(range(len(MACHINES)))
ax1.set_yticklabels(MACHINE_LABELS)
ax1.set_xlabel('経過時間（秒）')
ax1.set_title('チョコ停発生タイムライン（太いバー = 先頭ワーク・停止時間）')

# ── 棒グラフ：設備ごとの回数と停止時間 ──
summary = (
    leaders_df.groupby('machine')
    .agg(count=('work_id', 'count'),
         total_min=('duration_sec', lambda x: x.sum() / 60))
    .reindex(MACHINES).fillna(0)
)

x, width = np.arange(len(MACHINES)), 0.35
ax2r = ax2.twinx()
ax2.bar(x - width / 2, summary['count'],     width, color=COLORS, alpha=0.85, label='チョコ停回数')
ax2r.bar(x + width / 2, summary['total_min'], width, color=COLORS, alpha=0.45, label='停止時間合計（分）')

ax2.set_xticks(x)
ax2.set_xticklabels(MACHINE_LABELS)
ax2.set_ylabel('チョコ停回数（回）')
ax2r.set_ylabel('停止時間合計（分）')
ax2.set_title('設備別チョコ停実績（先頭ワーク帰属）')

plt.tight_layout()
plt.savefig('samples/choco_stop_analysis.png', dpi=150, bbox_inches='tight')
```

出力されるグラフが下の画像だ。

![チョコ停分析グラフ](https://static.zenn.studio/user-upload/c5e4362d53a0-20260518.png)

上段のガントチャートでは、各設備の通過リズムを細いバーで示し、チョコ停（先頭ワーク）を太いバーで強調している。プレスでW-015・W-035が、研磨でW-026が、検査でW-042が止まっていることが一目でわかる。

下段の棒グラフでは、設備ごとにチョコ停の回数（濃いバー）と停止時間合計（薄いバー）を並べている。回数が多い設備と停止時間が長い設備は必ずしも一致しない。「回数は少ないが一度止まると長い」設備が浮かび上がることがある。

---

## 現場に適用するときのポイント

| ポイント | 内容 |
|---|---|
| 標準タクトの推定 | ログの中央値を使う。平均は外れ値（チョコ停時のタクト）に引っ張られるため不向き |
| 閾値の決め方 | まず 1.5 倍で試し、現場の「明らかに止まった」感覚と照らし合わせて調整する。誤検出が多ければ 2.0 倍に上げる |
| DB 接続への差し替え | `pd.read_csv` を `pd.read_sql` に変えるだけ。それ以外のコードは変わらない |
| ログの粒度 | 設備入口のセンサー信号をトリガーにしたタイムスタンプが理想。工程完了信号でも代用できる |

---

## 実装上の課題と対応アイデア

**課題1：複数設備が同時刻にチョコ停した**

- まず試すこと：どちらの遅れが先に発生したかをタイムスタンプで比較する。設備間の伝播時間（標準処理時間）を考慮すれば、どちらが起点かを推定できる
- それでもダメなら：両方を先頭候補として記録し、現場担当者に確認を促すフラグを立てる

**課題2：データ欠損がある（通過しなかったワークの記録がない）**

- まず試すこと：欠損行を `.dropna()` で除外してからタクト計算する。ただしタクトの連続性が崩れるため、欠損の前後には注意が必要
- 根本対策：欠損そのものを「設備停止イベント」として別途記録する仕組みを設備側に追加する

**課題3：設備ごとに標準タクトが大きく異なる**

- 標準タクトを設備ごとに辞書で管理するように変更する。本記事のコードでは `standard_tacts` が設備ごとの辞書になっているので、固定値に上書きするだけで対応できる

---

## まとめ

| 処理 | 使う技術 |
|---|---|
| タクト計算 | `pandas` `.diff()` |
| 先頭判定 | `.shift()` による前行比較 |
| 停止時間推定 | 先頭から連続するタクト遅れの積算 |
| 上流帰属フィルタ | `work_id` の重複チェック |
| 可視化 | `matplotlib` ガントチャート＋棒グラフ |

チョコ停の原因設備を正しく特定するには、「タクトが遅かった設備」ではなく「最初に止まった設備」を見つけなければならない。その判定を、画像処理なし・pandas の `.diff()` と `.shift()` だけで実装できる——それが今回伝えたかったことだ。

---

## GitHubリポジトリ

サンプルコード・サンプルデータ一式を公開しています。

🔗 [github.com/factory-dx-eng/manufacturing-python](https://github.com/factory-dx-eng/manufacturing-python/tree/main/choco_stop_detection)
