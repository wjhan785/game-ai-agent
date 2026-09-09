#!/usr/bin/env bash
# Installs a pre-commit hook that blocks a commit containing anything
# shaped like an API key. .git/hooks/ is not versioned, so this script
# (which IS versioned) is what actually puts the hook in place -- run it
# once after cloning:
#
#   bash scripts/install-hooks.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit"

cat > "$HOOK_PATH" << 'HOOK'
#!/usr/bin/env bash
# Blocks any commit whose staged content contains something shaped like
# an API key. See tests/test_no_secrets.py for the same pattern applied
# to the whole working tree.
set -euo pipefail

pattern='sk-[A-Za-z0-9]{16,}'
staged_hits="$(git diff --cached -U0 | grep -E "^\+" | grep -Ev '^\+\+\+' | grep -E "$pattern" || true)"

if [ -n "$staged_hits" ]; then
    echo "pre-commit: refusing to commit -- staged content looks like an API key:" >&2
    echo "$staged_hits" >&2
    echo "If this is a false positive, use 'git commit --no-verify' deliberately." >&2
    exit 1
fi
HOOK

chmod +x "$HOOK_PATH"
echo "Installed pre-commit hook at $HOOK_PATH"
