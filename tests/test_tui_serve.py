"""The Serve dialog in the app: start, call the API, close = stopped."""
import asyncio
import json
import urllib.request

import pytest

pytest.importorskip("fastapi")
from aiforge.cli import main  # noqa: E402
from aiforge.serve import port_free  # noqa: E402
from aiforge.tui.app import AIForgeApp  # noqa: E402

PORT = 8799


def test_serve_dialog_starts_and_stops(tmp_path):
    assert main(["-w", str(tmp_path), "new", "spiral", "-t", "1"]) == 0
    assert main(["-w", str(tmp_path), "train", "spiral", "-d", "spirals", "-e", "1", "--device", "cpu"]) == 0

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.press("a")
            dialog = app.screen
            dialog.query_one("#port").value = str(PORT)
            await pilot.click("#ok")
            for _ in range(100):
                await pilot.pause(0.05)
                if dialog.server:
                    break
            assert dialog.server, dialog.query_one("#error").render()
            info = await asyncio.to_thread(
                lambda: json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/info")))
            assert info["model"] == "spiral"
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert app.screen is not dialog
        assert port_free("127.0.0.1", PORT)

    asyncio.run(scenario())


def test_install_hint_shows_the_extra(tmp_path, monkeypatch):
    # "[serve]" must reach the screen: Textual would otherwise read it as a style tag and drop it.
    import aiforge.serve

    monkeypatch.setattr(aiforge.serve, "missing_dependencies", lambda: aiforge.serve.INSTALL_HINT)
    assert main(["-w", str(tmp_path), "new", "spiral", "-t", "1"]) == 0
    assert main(["-w", str(tmp_path), "train", "spiral", "-d", "spirals", "-e", "1", "--device", "cpu"]) == 0

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.press("a")
            await pilot.pause(0.1)
            shown = str(app.screen.query_one("#error").render())
            assert '"aiforge-studio[serve]"' in shown
            assert app.screen.query_one("#ok").disabled

    asyncio.run(scenario())
