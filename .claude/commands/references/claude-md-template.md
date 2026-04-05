# CLAUDE.md テンプレート（秘書設定用）

プロジェクトの CLAUDE.md に追記する秘書関連の設定テンプレート。

---

## 追記例

```markdown
## Secretary設定

このプロジェクトでは `.secretary/` フォルダでタスク・アイデア・リサーチを管理しています。

### `/secretary` コマンドの使い方

| コマンド | 動作 |
|---|---|
| `/secretary` | 管理モード起動（メニュー表示） |
| `/secretary タスク追加 [内容]` | 今日のタスクに追加 |
| `/secretary 今日のタスク` | 今日のタスクを表示 |
| `/secretary メモ [内容]` | inboxにキャプチャ |
| `/secretary アイデア [タイトル]` | アイデアファイルを作成 |
| `/secretary 調査 [タイトル]` | リサーチファイルを作成 |
| `/secretary 週次レビュー` | 週次レビューを生成 |
| `/secretary ダッシュボード` | 全体概要を表示 |

### 管理カテゴリ

（インストール時に選択したカテゴリを記載）
```

---

## config.json スキーマ

```json
{
  "name": "string（オプション）",
  "role": "string — ユーザーの役割・職業",
  "categories": ["string — 選択されたカテゴリ名の配列"],
  "created_at": "YYYY-MM-DD",
  "version": "1.0"
}
```

## カテゴリ名一覧

```
todos
ideas
research
knowledge
content-plan
meetings
clients
journal
reading-list
debugging
projects
finances
inbox        ← 常に含む
reviews      ← 常に含む
```
