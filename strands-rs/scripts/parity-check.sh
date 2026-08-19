#!/usr/bin/env bash
#
# parity-check.sh — report strands-rs files whose strands-ts source drifted.
#
# Reads docs/parity-manifest.toml and, for each entry, lists strands-ts commits
# that touched the mapped `ts_source` since the entry's pinned `upstream_commit`.
# Any such commit means the Rust file should be reviewed for parity. Also flags
# manifest rot: entries whose `rust` file is gone, and Rust files carrying a
# `Ports ` doc comment that are missing from the manifest.
#
# Usage:
#   scripts/parity-check.sh          # advisory: print a report, always exit 0
#   scripts/parity-check.sh --strict # exit non-zero if any drift/rot is found
#
# strands-ts lives in the same monorepo, so "upstream" commits are monorepo
# commits that touched the strands-ts/ subtree.

set -euo pipefail

STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(dirname "$SCRIPT_DIR")"
MANIFEST="$CRATE_DIR/docs/parity-manifest.toml"
HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"

if [ ! -f "$MANIFEST" ]; then
  echo "parity-check: manifest not found at $MANIFEST" >&2
  exit 2
fi

echo "parity-check: manifest=$MANIFEST"
echo "parity-check: monorepo HEAD=$HEAD_SHA"
echo

drift=0
rot=0

# Extract rust|ts_source|upstream_commit for each [[entry]] block.
entries="$(awk '
  /^\[\[entry\]\]/ { if (have) print rust "|" ts "|" commit; have=1; rust=""; ts=""; commit=""; next }
  /^rust = / { v=$0; sub(/^rust = "/,"",v); sub(/".*$/,"",v); rust=v; next }
  /^ts_source = / { v=$0; sub(/^ts_source = "/,"",v); sub(/".*$/,"",v); ts=v; next }
  /^upstream_commit = / { v=$0; sub(/^upstream_commit = "/,"",v); sub(/".*$/,"",v); commit=v; next }
  END { if (have) print rust "|" ts "|" commit }
' "$MANIFEST")"

# Track which rust files are in the manifest (for the orphan check).
manifest_rust="$(printf '%s\n' "$entries" | cut -d'|' -f1 | sort -u)"

echo "== Upstream drift since pinned commits =="
while IFS='|' read -r rust ts commit; do
  [ -z "$rust" ] && continue

  # Manifest rot: mapped Rust file no longer exists.
  if [ ! -e "$REPO_ROOT/$rust" ]; then
    echo "  [ROT]   $rust — mapped Rust file does not exist"
    rot=$((rot + 1))
    continue
  fi

  # No upstream source to compare (Rust-only glue / Python-sourced construct).
  [ -z "$ts" ] && continue

  if [ ! -e "$REPO_ROOT/$ts" ]; then
    echo "  [ROT]   $ts — ts_source path does not exist (for $rust)"
    rot=$((rot + 1))
    continue
  fi

  # Commits touching the ts_source since the pinned commit.
  log="$(git -C "$REPO_ROOT" log --oneline "$commit..$HEAD_SHA" -- "$ts" 2>/dev/null || true)"
  if [ -n "$log" ]; then
    count="$(printf '%s\n' "$log" | grep -c . || true)"
    echo "  [DRIFT] $rust  <=  $ts  ($count upstream commit(s) since ${commit:0:12})"
    printf '%s\n' "$log" | sed 's/^/            /'
    drift=$((drift + 1))
  fi
done <<< "$entries"
[ "$drift" -eq 0 ] && echo "  (none — all pinned sources unchanged)"

echo
echo "== Manifest coverage (Rust files with a \`Ports\` comment) =="
# Rust files that claim to port something but aren't in the manifest.
orphans=0
while IFS= read -r file; do
  rel="${file#"$REPO_ROOT"/}"
  if ! printf '%s\n' "$manifest_rust" | grep -qxF "$rel"; then
    echo "  [ORPHAN] $rel — has a \`Ports\` doc comment but is not in the manifest"
    orphans=$((orphans + 1))
  fi
done < <(grep -rlE "[Pp]orts \`" "$CRATE_DIR/strands/src" "$CRATE_DIR/strands-macros/src" --include='*.rs' 2>/dev/null | sort)
[ "$orphans" -eq 0 ] && echo "  (none — every ported file is mapped)"
rot=$((rot + orphans))

echo
echo "parity-check: $drift file(s) drifted, $rot manifest issue(s)."
if [ "$STRICT" -eq 1 ] && { [ "$drift" -ne 0 ] || [ "$rot" -ne 0 ]; }; then
  exit 1
fi
exit 0
