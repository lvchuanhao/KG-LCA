"""Expose settings at package level for convenient import

This allows existing code that does `from config import VAR` to keep working
by re-exporting symbols defined in `config.settings`.
"""
from importlib import import_module
import sys as _sys

_settings = import_module("config.settings")
_globals = globals()

# Re-export public attributes defined in settings
for _name in dir(_settings):
    if _name.startswith("__"):
        continue
    _globals[_name] = getattr(_settings, _name)

# Make sure modules referencing `config` resolve to this package
_sys.modules.setdefault("config", _sys.modules[__name__])
