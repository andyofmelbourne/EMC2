#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Unit tests ==="
pytest test_unit.py -v

echo ""
echo "=== Integration tests (CPU logR + probability) ==="
pytest test_integration.py -v

echo ""
echo "=== Full pipeline (GPU + MPI) ==="
python run_emc_maxwell.py
