#!/bin/sh
# Use the isolated toolchain without editing the user's shell profile.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export OPAMROOT="$ROOT/.tools/opam-root"
export SDKROOT="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk}"
export PATH="$ROOT/.tools/rocq-deps/bin:$PATH"
export PKG_CONFIG_PATH="$ROOT/.tools/rocq-deps/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
exec "$ROOT/.tools/opam-2.6.0-arm64-macos" exec --switch=ojasx-rocq-9.2 -- "$@"
