"""Theme-aware colours for widgets that draw Rich text themselves (charts, bars, the banner).

CSS can use ``$primary`` etc. directly; Rich styles and markup can't, so they ask here.
Values are hex strings from the current Textual theme, so they suit dark and light themes.
"""
from __future__ import annotations

DARK, LIGHT = "textual-dark", "textual-light"
_FALLBACK = {"primary": "cyan", "accent": "magenta", "success": "green", "warning": "yellow",
             "error": "red", "foreground": "white"}


def tone(node, name: str) -> str:
    """Colour ``name`` (primary, accent, success, warning, error, foreground) for ``node``'s app."""
    try:
        return node.app.theme_variables.get(name) or _FALLBACK[name]
    except Exception:  # no running app (e.g. tests)
        return _FALLBACK[name]


def is_dark(node) -> bool:
    try:
        return node.app.current_theme.dark
    except Exception:
        return True
