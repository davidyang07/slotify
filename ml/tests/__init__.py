"""Test package marker.

Present so ``tests.dataset_fixtures`` is importable from every test module: with
an ``__init__.py`` here, pytest's ``prepend`` import mode puts ``ml/`` on
``sys.path`` rather than ``ml/tests/``.
"""
