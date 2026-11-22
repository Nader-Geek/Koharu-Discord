#!/usr/bin/env bash
# Builds koharu-headless, Koharu's windowless worker, from koharu-rs/koharu at
# the commit UPSTREAM_COMMIT pins, with this folder's crate added to its
# workspace and its patches applied. The binary lands in ../out/.
#
#   bash worker/build.sh                               # clones into .build/koharu
#   KOHARU_SRC=/path/to/koharu bash worker/build.sh    # an existing checkout
#   bash worker/build.sh -j 2                          # extra cargo arguments
#
# Needs: git, a Rust toolchain (1.97.1 or later), clang and libclang for
# bindgen, pkg-config, and OpenSSL headers.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
UPSTREAM="${KOHARU_UPSTREAM:-https://github.com/koharu-rs/koharu.git}"
PIN="$(tr -d '[:space:]' < "$HERE/UPSTREAM_COMMIT")"
OUT="${KOHARU_OUT:-$ROOT/out}"

if [ -n "${KOHARU_SRC:-}" ]; then
  # Someone's checkout: used as it is, never fetched or cleaned.
  SRC="$KOHARU_SRC"
  if [ ! -d "$SRC/.git" ]; then
    echo "KOHARU_SRC=$SRC isn't a git checkout" >&2
    exit 1
  fi
  head="$(git -C "$SRC" rev-parse HEAD)"
  if [ "$head" != "$PIN" ]; then
    echo "warning: $SRC is at $head, not the pinned $PIN the crate was written against" >&2
  fi
else
  # The script's own clone, kept at the pin.
  SRC="$ROOT/.build/koharu"
  if [ ! -d "$SRC/.git" ]; then
    echo "cloning koharu-rs/koharu at $PIN into $SRC"
    mkdir -p "$SRC"
    git -C "$SRC" init -q
    git -C "$SRC" remote add origin "$UPSTREAM"
  fi
  if [ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)" != "$PIN" ]; then
    git -C "$SRC" fetch -q --depth 1 origin "$PIN"
    git -C "$SRC" checkout -q --force --detach FETCH_HEAD
  fi
fi

echo "adding crates/koharu-headless"
rm -rf "$SRC/crates/koharu-headless"
cp -R "$HERE/koharu-headless" "$SRC/crates/koharu-headless"
for patch in "$HERE"/patches/*.patch; do
  if git -C "$SRC" apply --check --reverse "$patch" 2>/dev/null; then
    echo "$(basename "$patch") is already applied"
    continue
  fi
  echo "applying $(basename "$patch")"
  git -C "$SRC" apply "$patch"
done

# Debug info is for Sentry's stack traces, which this worker doesn't send;
# without it the link needs far less memory and the binary is a third the size.
export CARGO_PROFILE_RELEASE_DEBUG="${CARGO_PROFILE_RELEASE_DEBUG:-0}"
(cd "$SRC" && cargo build --release -p koharu-headless "$@")

mkdir -p "$OUT"
cp "$SRC/target/release/koharu-headless" "$OUT/koharu-headless"
strip "$OUT/koharu-headless" 2>/dev/null || true
echo "built $OUT/koharu-headless"
"$OUT/koharu-headless" --version
