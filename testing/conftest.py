import os
import sys
from pathlib import Path

TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent))


def pytest_configure(config):
    """Run all tests from the testing/ directory (needed for relative data paths)."""
    os.chdir(TEST_DIR)
    config.addinivalue_line(
        "filterwarnings", "ignore::pyopencl.CompilerWarning"
    )
