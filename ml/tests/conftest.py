"""Shared pytest setup.

Workaround for a machine-specific defect, not a project requirement: pytest
creates its ``tmp_path`` fixtures under ``$TMPDIR/pytest-of-$USER``, and on this
developer's Windows install that directory exists but denies enumeration
(``PermissionError: [WinError 5]``), which errors out every test using
``tmp_path`` before it runs. Rather than changing ACLs on a system directory, we
redirect pytest's temp root into the project when -- and only when -- the
default root is unusable. On a healthy machine this is a no-op.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _default_temp_root_is_usable() -> bool:
    root = Path(tempfile.gettempdir()) / f"pytest-of-{os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))}"
    if not root.exists():
        return True  # pytest will create it
    try:
        next(os.scandir(root), None)
    except OSError:
        return False
    return True


if not _default_temp_root_is_usable():
    fallback = Path(__file__).resolve().parents[1] / ".pytest-tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(fallback)
