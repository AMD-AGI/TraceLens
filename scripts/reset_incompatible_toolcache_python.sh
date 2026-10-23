#!/bin/sh
# Drop a host toolcache Python before actions/setup-python runs in this container.
#
# actions/setup-python reuses the runner toolcache without checking that the
# build matches the container OS. GitHub mounts that cache at /__t. An Ubuntu
# 26 CPython then fails inside the Ubuntu 24.04 image with "ssl module is not
# available", and pip cannot reach PyPI. Removing the requested version makes
# setup-python download the build for this container.
#
# Do not probe ``import ssl`` here: it can succeed before apt updates the
# container's libraries and fail afterwards, which is exactly the failure this
# script prevents.

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

[ -d "$root" ] || exit 0

version=${TOOLCACHE_PYTHON_VERSION:-3.10}

find "$root" -mindepth 1 -maxdepth 1 -type d -name "$version.*" -print -exec rm -rf {} +
