#!/bin/bash
# git push 後に published: true の記事が含まれていたら /zenn-review の実行を促す

input=$(cat)
cmd=$(echo "$input" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('command', ''))
except:
    print('')
" 2>/dev/null)

if echo "$cmd" | grep -q "git push"; then
    repo_dir="$(git -C /Users/yuya.niizuma/pywork/manufacturing-python rev-parse --show-toplevel 2>/dev/null)"
    changed=$(git -C "$repo_dir" diff HEAD~1 HEAD -- 'articles/*.md' 2>/dev/null | grep '^+published: true')
    if [ -n "$changed" ]; then
        echo ""
        echo "📢 記事が公開されました！"
        echo "→ /zenn-review を実行してX投稿スレッド草案をNotionに保存しましょう"
    fi
fi
