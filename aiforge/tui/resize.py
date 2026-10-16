"""A draggable edge for resizing a right-docked side panel."""
from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widget import Widget

MIN_FRACTION, MAX_FRACTION, DEFAULT_FRACTION = 0.25, 0.85, 0.46


class ResizeHandle(Widget):
    """Sits on the left edge of a right-docked panel. Drag it to change the panel width.

    Posts ``ResizeHandle.Resized(fraction)`` while dragging and ``Reset`` on double-click.
    """

    DEFAULT_CSS = """
    ResizeHandle { dock: left; width: 1; height: 100%; background: $accent 40%; }
    ResizeHandle:hover, ResizeHandle.-dragging { background: $accent; }
    """

    class Resized(Message):
        def __init__(self, fraction: float, done: bool) -> None:
            super().__init__()
            self.fraction, self.done = fraction, done

    class Reset(Message):
        pass

    def __init__(self, **kw):
        super().__init__(**kw)
        self.tooltip = "Drag to resize · double-click to reset"
        self._dragging = False

    def render(self) -> str:
        h = self.size.height
        return "\n".join("⋮" if abs(i - h // 2) <= 1 else " " for i in range(h))

    def _fraction(self, screen_x: int) -> float:
        width = self.screen.size.width or 1
        return min(MAX_FRACTION, max(MIN_FRACTION, (width - screen_x) / width))

    def on_click(self, event: events.Click) -> None:
        if event.chain >= 2:  # double-click
            self.post_message(self.Reset())

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._dragging = True
        self.add_class("-dragging")
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging:
            self.post_message(self.Resized(self._fraction(event.screen_x), done=False))

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.remove_class("-dragging")
            self.release_mouse()
            self.post_message(self.Resized(self._fraction(event.screen_x), done=True))
