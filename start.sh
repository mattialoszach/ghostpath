#!/bin/sh
set -eu
cd "$(dirname "$0")"

extra=record
if [ "${1:-}" = "--hardware" ]; then
    extra=hardware
    shift
fi

if ! command -v uv >/dev/null || ! command -v npm >/dev/null; then
    echo "Install uv and Node.js 22.12+ (including npm), then run this script again." >&2
    exit 1
fi

if [ ! -d frontend/node_modules ] || [ frontend/package-lock.json -nt frontend/node_modules/.package-lock.json ]; then
    npm --prefix frontend ci
fi
npm --prefix frontend run build
exec uv run --locked --python 3.12 --extra "$extra" ghostpath "$@"
