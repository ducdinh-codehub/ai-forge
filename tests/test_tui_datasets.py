"""Dataset hub: a download only ever starts from the Download button."""
import asyncio

from textual.widgets import Button, OptionList

from aiforge.tui.app import AIForgeApp
from aiforge.tui.datasets import DatasetScreen


def test_click_on_name_does_not_download(tmp_path):
    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await app.push_screen(DatasetScreen())
            await pilot.pause(0.2)
            screen, started = app.screen, []
            screen._start_download = lambda: started.append(screen.current.key)
            lst = screen.query_one("#hub-list", OptionList)
            for y in range(lst.region.y, lst.region.y + 8):  # click every visible name, twice
                await pilot.click(offset=(lst.region.x + 5, y))
                await pilot.click(offset=(lst.region.x + 5, y))
            lst.focus()
            await pilot.press("enter")  # Enter on a name: also no download yet
            await pilot.pause(0.1)
            assert started == []
            assert app.focused is screen.query_one("#download", Button)  # ready to press
            await pilot.click("#download")
            await pilot.pause(0.1)
            assert started == [screen.current.key]

    asyncio.run(scenario())
