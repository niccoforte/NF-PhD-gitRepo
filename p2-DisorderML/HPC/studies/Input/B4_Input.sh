#!/bin/bash
# Preview by default; only --submit queues the Input study.
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "Usage: $0 SUITE [--submit]"; exit 2; fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
suite=$1; shift
exec bash "$script_dir/../B4_Field-source-study.sh" "$suite" inputs "$@"
