
#!/usr/bin/env python3

"""Portable repository-root resolution for NuCLS reproducibility scripts."""



from __future__ import annotations



import os

from pathlib import Path





def repo_root() -> Path:

    """Return NUCLS repository root.



    Priority:

    1. NUCLS_PROJECT_ROOT environment variable

    2. Repository location inferred from this file

    """

    env = os.environ.get("NUCLS_PROJECT_ROOT")

    if env:

        return Path(env).expanduser().resolve()



    return Path(__file__).resolve().parent.parent

