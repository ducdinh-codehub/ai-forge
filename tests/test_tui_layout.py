"""The Home screen fits the smallest supported terminal (120 × 35)."""
import asyncio

from rich.cells import cell_len

from aiforge.tui.app import AIForgeApp


def test_home_actions_fit_two_rows(tmp_path):
    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(120, 35)) as pilot:
            await pilot.pause(0.3)
            buttons = list(app.screen.query("#details-pane > .toolbar Button"))
            assert len({b.region.y for b in buttons}) == 2
            cut = [b.id for b in buttons if cell_len(str(b.label)) > b.content_region.width]
            assert not cut, f"labels cut off: {cut}"

    asyncio.run(scenario())
