#!/bin/bash
# Preview by default; selected source/reused banks are explicit environment inputs.
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "Usage: $0 SUITE [--submit]"; exit 2; fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
suite=$1; shift
exec bash "$script_dir/../B4_Field-source-study.sh" "$suite" losses "$@"
