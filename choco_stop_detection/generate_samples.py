"""
連続ライン ワーク通過タイムスタンプログ生成
チョコ停を人工的に注入して、分析コードの動作確認用サンプルを作る
"""
import os
import pandas as pd
from datetime import datetime, timedelta

MACHINES = ['furnace', 'press', 'polish', 'inspect', 'pack']

# 各設備の標準処理時間（秒）
PROC_TIMES = {
    'furnace': 120,
    'press':    60,
    'polish':   90,
    'inspect':  45,
    'pack':     30,
}

STANDARD_TACT = 30  # ライン投入間隔（秒）
N_WORKS = 60

# チョコ停イベント: (設備, 先頭ワークindex, 停止時間[秒])
STOP_EVENTS = [
    ('press',   14,  90),   # press  W-015 を先頭に90秒停止
    ('press',   34,  60),   # press  W-035 を先頭に60秒停止
    ('inspect', 41, 120),   # inspect W-042 を先頭に120秒停止
    ('polish',  25,  75),   # polish W-026 を先頭に75秒停止
]


def generate_log():
    stop_dict = {}
    for machine, work_idx, duration in STOP_EVENTS:
        stop_dict.setdefault(machine, {})[work_idx] = duration

    base_time = datetime(2024, 3, 15, 8, 0, 0)
    entries = {}

    # furnace への投入は標準タクトで固定
    entries['furnace'] = [
        base_time + timedelta(seconds=i * STANDARD_TACT) for i in range(N_WORKS)
    ]

    # 上流設備の処理完了 → 下流設備へ流入、設備の空き待ちをシミュレート
    for m_idx in range(1, len(MACHINES)):
        machine   = MACHINES[m_idx]
        prev      = MACHINES[m_idx - 1]
        prev_proc = PROC_TIMES[prev]

        machine_free_at  = datetime(2000, 1, 1)  # 開始時は即利用可
        machine_entries  = []
        stops = stop_dict.get(machine, {})

        for i in range(N_WORKS):
            work_arrives = entries[prev][i] + timedelta(seconds=prev_proc)
            entry_time   = max(work_arrives, machine_free_at)

            # チョコ停（先頭ワーク）の場合は停止時間を加算
            if i in stops:
                entry_time += timedelta(seconds=stops[i])

            machine_entries.append(entry_time)
            machine_free_at = entry_time + timedelta(seconds=PROC_TIMES[machine])

        entries[machine] = machine_entries

    records = [
        {'work_id': f'W-{i+1:03d}', **{f'{m}_in_time': entries[m][i] for m in MACHINES}}
        for i in range(N_WORKS)
    ]
    return pd.DataFrame(records)


if __name__ == '__main__':
    os.makedirs('samples', exist_ok=True)
    df = generate_log()
    df.to_csv('samples/choco_stop_log.csv', index=False)
    print(df.head(20).to_string())
    print(f'\n{len(df)} works generated → samples/choco_stop_log.csv')
