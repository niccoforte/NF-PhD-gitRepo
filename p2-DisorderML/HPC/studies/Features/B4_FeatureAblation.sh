#!/bin/bash
# CNN2/MSE context attribution; preserve operational presence masks.
entry=studies/Features/A0-FeatureAblation.py
plans=(
    "control|UT|42" "no_xy|UT|42" "no_designable|UT|42"
    "no_interfaces|UT|42" "no_present|UT|42" "no_context|UT|42"
    "control|FT|42" "no_xy|FT|42" "no_designable|FT|42"
    "no_pins|FT|42" "no_tip|FT|42" "no_tip_offsets|FT|42"
    "no_tip_distance|FT|42" "no_present|FT|42" "no_context|FT|42"
)
source "$(dirname "${BASH_SOURCE[0]}")/../B4_Final-field-study.sh" "$@"
