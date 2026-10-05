#!/usr/bin/env bash
# Publish step of the PR-review workflow: turn the review's card.md into a PR
# verdict. Request changes when the card reports blocking issues, otherwise post
# an LGTM comment. Never approve, never merge -- that is the human reviewer's call.
#
# usage: pr-review-publish.sh <PR-NUMBER> <CARD-FILE> <MARKER>

set -euo pipefail

PR="${1:?usage: pr-review-publish.sh <PR> <card.md> <marker>}"
CARD="${2:?missing card file}"
MARKER="${3:-<!-- tracelens-pr-review -->}"
REPO="${REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"

[ -s "$CARD" ] || { echo "pr-review-publish: $CARD is empty or missing" >&2; exit 1; }

# The review writes exactly one "Blocking issues: <N|none>" line. Read the count
# from it; anything but an explicit zero/none is treated as blocking.
blocking_line="$(grep -iE '^Blocking issues:' "$CARD" | head -1 || true)"
[ -n "$blocking_line" ] || { echo "pr-review-publish: card has no 'Blocking issues:' line" >&2; exit 1; }

count="$(printf '%s' "$blocking_line" | sed -E 's/^[Bb]locking issues:[[:space:]]*//')"

is_clean=0
case "$(printf '%s' "$count" | tr '[:upper:]' '[:lower:]')" in
  none | 0 | "none.") is_clean=1 ;;
esac

# Carry the marker (for the 2-review cap) plus the card body as the comment.
body="$(printf '%s\n\n%s' "$MARKER" "$(cat "$CARD")")"

if [ "$is_clean" = 1 ]; then
  gh pr comment "$PR" --repo "$REPO" --body "$body"$'\n\nLGTM'
else
  gh pr review "$PR" --repo "$REPO" --request-changes --body "$body"
fi
