#!/bin/bash
# CNN2+spatial three-seed checks and missing CNN3 depth confirmations.
entry=studies/Locality/A0-CNNSpatial.py
plans=(
    "cnn2_spatial|UT|42" "cnn2_spatial|FT|42"
    "cnn2_spatial|UT|43" "cnn2_spatial|FT|43"
    "cnn2_spatial|UT|44" "cnn2_spatial|FT|44"
    "cnn3|UT|43" "cnn3|FT|43" "cnn3|UT|44" "cnn3|FT|44"
)
source "$(dirname "${BASH_SOURCE[0]}")/../B4_Final-field-study.sh" "$@"
