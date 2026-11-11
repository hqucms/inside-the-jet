"""Run the command line ``jetgen`` as ``python -m itj.toy``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
