#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "============================================================"
echo " OjasX3 + Aegis + OjasX-LogicIR (Phase 5) Unified Verifier"
echo " Host: $(uname -s) $(uname -m) | Python: $(python3 --version)"
echo "============================================================"

echo "[1/4] Checking Python syntax across ojasx2, ojasx3, ojasx_safety, ojasx_logic, scripts..."
PYTHONPATH="$ROOT_DIR" python3 -m compileall -q ojasx2 ojasx3 ojasx_safety ojasx_logic scripts tests/test_ojasx_logic_compiler.py
echo "      OK"

echo "[2/4] Running OjasX-LogicIR + Zero-Xcode Metal 4 Runtime + Z3 + IREE Suite..."
if python3 -m pytest --version >/dev/null 2>&1; then
  PYTHONPATH="$ROOT_DIR" OJASX_STRESS=1 python3 -m pytest tests/test_ojasx_logic_compiler.py -v -s
else
  PYTHONPATH="$ROOT_DIR" OJASX_STRESS=1 python3 tests/test_ojasx_logic_compiler.py
fi

if python3 -c "import torch" >/dev/null 2>&1 && python3 -m pytest --version >/dev/null 2>&1; then
  echo "[3/4] Running OjasX3 Core + Aegis Red-Team Hardening Suites..."
  PYTHONPATH="$ROOT_DIR" python3 -m pytest \
    tests/test_aegis_redteam_hardening.py \
    tests/test_production_readiness_v2.py \
    tests/test_ojasx3_production_upgrade.py \
    tests/test_upgrade_edge_cases.py \
    -v

  echo "[4/4] Running RMSNorm synthetic9b 9.0B finite-loss check + 1.5B Metal shader check..."
  PYTHONPATH="$ROOT_DIR" python3 scripts/run_hf_9b_finetune_bench.py --preset synthetic9b --steps 3 --seq-len 128 --json-out bench_synthetic9b_rmsnorm.json
else
  echo "[3/4] Skipped PyTorch/Aegis pytest suites (torch/pytest not installed on this host)"
  echo "[4/4] Skipped synthetic9b MPS benchmark (torch not installed on this host)"
fi

echo "============================================================"
echo " Unified verification complete."
echo "============================================================"
