#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

g++ -O3 -std=c++17 -pthread \
  "$SCRIPT_DIR/hsp2_ibs_fast.cpp" \
  -o "$SCRIPT_DIR/hsp2_ibs_fast"

g++ -O3 -std=c++17 -pthread \
  "$SCRIPT_DIR/tree_simulate.cpp" \
  -o "$SCRIPT_DIR/tree_simulate"

g++ -O3 -std=c++17 -pthread \
  "$SCRIPT_DIR/eflop_simulate.cpp" \
  -o "$SCRIPT_DIR/eflop_simulate"

g++ -O3 -std=c++17 -pthread \
  "$SCRIPT_DIR/cpp_model_recovery.cpp" \
  -o "$SCRIPT_DIR/cpp_model_recovery"

echo "Built $SCRIPT_DIR/hsp2_ibs_fast"
echo "Built $SCRIPT_DIR/tree_simulate"
echo "Built $SCRIPT_DIR/eflop_simulate"
echo "Built $SCRIPT_DIR/cpp_model_recovery"
