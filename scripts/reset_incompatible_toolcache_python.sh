#!/bin/sh
# Drop host toolcache Pythons that cannot import ssl in this container.
#
# actions/setup-python reuses the runner toolcache without checking that the
# build matches the container OS. GitHub mounts that cache at /__t. An Ubuntu
# 26 CPython then fails inside the Ubuntu 24.04 image with "ssl module is not
# available", and pip cannot reach PyPI. Removing the broken install makes
# setup-python download the build for this container.

set -eu

if [ -n "${TOOLCACHE_PYTHON_ROOT:-}" ]; then
  root=$TOOLCACHE_PYTHON_ROOT
elif [ -d /__t/Python ]; then
  root=/__t/Python
elif [ -n "${RUNNER_TOOL_CACHE:-}" ] && [ -d "$RUNNER_TOOL_CACHE/Python" ]; then
  root=$RUNNER_TOOL_CACHE/Python
else
  exit 0
fi

candidates=$(mktemp)
find "$root" -type f \( -path '*/x64/bin/python' -o -path '*/x64/bin/python3' \) >"$candidates" 2>/dev/null || true

while IFS= read -r py; do
  [ -n "$py" ] && [ -x "$py" ] || continue
  prefix=$(CDPATH= cd "$(dirname "$py")/.." && pwd)
  version_dir=$(CDPATH= cd "$prefix/.." && pwd)
  [ -d "$version_dir" ] || continue
  if ! LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$py" -c 'import ssl' >/dev/null 2>&1; then
    echo "Removing toolcache Python that cannot import ssl: $version_dir"
    rm -rf "$version_dir"
  fi
done <"$candidates"

rm -f "$candidates"
