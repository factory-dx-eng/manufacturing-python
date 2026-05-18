"""
連続ライン チョコ停分析
・各設備のタクトを計算し、標準タクトを超えた先頭ワークを検出する
・先頭ワーク帰属でチョコ停回数・停止時間を集計して可視化する
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
import matplotlib.patches as mpatches

# ── パラメータ ────────────────────────────────────────────────
CSV_PATH       = 'samples/choco_stop_log.csv'
OUTPUT_PATH    = 'samples/choco_stop_analysis.png'
MACHINES       = ['furnace', 'press', 'polish', 'inspect', 'pack']
MACHINE_LABELS = ['① 炉', '② プレス', '③ 研磨', '④ 検査', '⑤ 梱包']
COLORS         = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']
TACT_THRESHOLD = 1.5   # 標準タクトの何倍以上をチョコ停と判定するか
# ─────────────────────────────────────────────────────────────


# ① データ読み込み
def load_data(path):
    df = pd.read_csv(path)
    for m in MACHINES:
        df[f'{m}_in_time'] = pd.to_datetime(df[f'{m}_in_time'])
    return df


# ② タクト計算（前ワークとの間隔）
def calc_tact(df):
    for m in MACHINES:
        df[f'{m}_tact'] = df[f'{m}_in_time'].diff().dt.total_seconds()
    return df


# ③ 先頭ワーク判定
def detect_leaders(df):
    """
    タクト遅れが発生し、かつ直前ワークは正常だったワークを「先頭」とする
    連続するタクト遅れは渋滞の波及なので先頭とは判定しない
    """
    standard_tacts = {
        m: df[f'{m}_tact'].median() for m in MACHINES
    }

    events = []
    for m in MACHINES:
        tact_col  = f'{m}_tact'
        threshold = standard_tacts[m] * TACT_THRESHOLD
        is_slow   = df[tact_col] > threshold
        is_leader = is_slow & ~is_slow.shift(1, fill_value=False)

        for idx in df[is_leader].index:
            # 先頭から連続するタクト遅れを集計して停止時間を推定
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
                'tact_sec':     df.loc[idx, tact_col],
                'delay_sec':    df.loc[idx, tact_col] - standard_tacts[m],
                'duration_sec': total_delay,
            })

    leaders = pd.DataFrame(events)
    if leaders.empty:
        return leaders, standard_tacts

    # 同一 work_id が上流設備でもすでに先頭判定されていれば下流側は除外
    # （上流チョコ停の波及を根本原因として帰属しない）
    upstream_leader_works = set()
    filtered = []
    for m in MACHINES:
        m_events = leaders[leaders['machine'] == m]
        for _, ev in m_events.iterrows():
            if ev['work_id'] not in upstream_leader_works:
                filtered.append(ev)
                upstream_leader_works.add(ev['work_id'])

    return pd.DataFrame(filtered), standard_tacts


# ④ 可視化
def plot_results(df, leaders_df):
    color_map = dict(zip(MACHINES, COLORS))
    label_map = dict(zip(MACHINES, MACHINE_LABELS))
    base_time = df['furnace_in_time'].min()

    def to_sec(t):
        return (t - base_time).total_seconds()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('連続ライン チョコ停分析', fontsize=15, fontweight='bold', y=0.98)

    # ── ガントチャート ────────────────────────────────────────
    for y, m in enumerate(MACHINES):
        # 全ワークの通過リズムを細いバーで表示
        for _, row in df.iterrows():
            ax1.barh(y, 20, left=to_sec(row[f'{m}_in_time']),
                     height=0.25, color=color_map[m], alpha=0.2)

        # チョコ停（先頭ワーク）を太いバーで強調
        for _, ev in leaders_df[leaders_df['machine'] == m].iterrows():
            dur = max(ev['duration_sec'], 40)
            ax1.barh(y, dur, left=to_sec(ev['entry_time']),
                     height=0.6, color=color_map[m], alpha=0.9)
            ax1.text(to_sec(ev['entry_time']) + dur / 2, y,
                     ev['work_id'], va='center', ha='center', fontsize=8,
                     color='black', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, lw=0))

    ax1.set_yticks(range(len(MACHINES)))
    ax1.set_yticklabels(MACHINE_LABELS, fontsize=11)
    ax1.set_xlabel('経過時間（秒）', fontsize=10)
    ax1.set_title('チョコ停発生タイムライン  （太いバー = 先頭ワーク・停止時間）', fontsize=11)
    ax1.grid(axis='x', alpha=0.3)

    legend_patches = [
        mpatches.Patch(color=color_map[m], label=label_map[m]) for m in MACHINES
    ]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=9)

    # ── 設備別集計 棒グラフ ───────────────────────────────────
    summary = (
        leaders_df.groupby('machine')
        .agg(count=('work_id', 'count'), total_min=('duration_sec', lambda x: x.sum() / 60))
        .reindex(MACHINES)
        .fillna(0)
    )

    x      = np.arange(len(MACHINES))
    width  = 0.35
    ax2r   = ax2.twinx()

    bars1 = ax2.bar(x - width / 2, summary['count'], width,
                    color=COLORS, alpha=0.85, label='チョコ停回数')
    bars2 = ax2r.bar(x + width / 2, summary['total_min'], width,
                     color=COLORS, alpha=0.45, label='停止時間合計（分）')

    for bar in bars1:
        h = bar.get_height()
        if h:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                     f'{int(h)}回', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        if h:
            ax2r.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                      f'{h:.1f}分', ha='center', va='bottom', fontsize=10)

    # ラベルがはみ出さないよう上限に余裕を持たせる
    max_count = summary['count'].max()
    max_min   = summary['total_min'].max()
    ax2.set_ylim(0, max_count * 1.4 if max_count else 1)
    ax2r.set_ylim(0, max_min * 1.4 if max_min else 1)

    ax2.set_xticks(x)
    ax2.set_xticklabels(MACHINE_LABELS, fontsize=11)
    ax2.set_ylabel('チョコ停回数（回）', fontsize=10)
    ax2r.set_ylabel('停止時間合計（分）', fontsize=10)
    ax2.set_title('設備別チョコ停実績（先頭ワーク帰属）', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f'Saved: {OUTPUT_PATH}')


def main():
    df           = load_data(CSV_PATH)
    df           = calc_tact(df)
    leaders_df, standard_tacts = detect_leaders(df)

    print('\n=== チョコ停先頭ワーク一覧 ===')
    cols = ['machine', 'work_id', 'entry_time', 'tact_sec', 'delay_sec', 'duration_sec']
    print(leaders_df[cols].to_string(index=False))

    plot_results(df, leaders_df)


if __name__ == '__main__':
    main()
