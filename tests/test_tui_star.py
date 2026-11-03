"""Asking for a GitHub star: once, after a successful training, never during the tour."""
import asyncio

from textual.screen import Screen

from aiforge import feedback
from aiforge.spec import Workspace, from_template
from aiforge.tui.app import AIForgeApp
from aiforge.tui.star import ASK_AGAIN_AFTER, StarDialog, count_training


def _app(tmp_path):
    Workspace(tmp_path).save_spec(from_template("MLP classifier (tabular)", "m"))  # no tour offer
    app = AIForgeApp(tmp_path)
    app.opened = []
    app.open_url = lambda url, **kw: app.opened.append(url)
    return app


async def _back_home(app, pilot):
    """Leave the home screen and come back, as after training."""
    await app.push_screen(Screen())
    app.pop_screen()
    await pilot.pause(0.2)


def test_asked_after_first_training_and_star_opens_github(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.3)
            await _back_home(app, pilot)
            assert not isinstance(app.screen, StarDialog)  # nothing trained yet
            count_training(app.ws)
            await _back_home(app, pilot)
            assert isinstance(app.screen, StarDialog)
            await pilot.click("#ok")
            await pilot.pause(0.2)
            assert app.opened == [feedback.REPO_URL]
            count_training(app.ws)
            await _back_home(app, pilot)
            assert not isinstance(app.screen, StarDialog)  # answered: never again

    asyncio.run(scenario())


def test_maybe_later_waits_and_dont_ask_again_ends_it(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.3)
            count_training(app.ws)
            await _back_home(app, pilot)
            await pilot.press("escape")  # = Maybe later
            await pilot.pause(0.2)
            assert not isinstance(app.screen, StarDialog)
            for _ in range(ASK_AGAIN_AFTER - 1):
                count_training(app.ws)
                await _back_home(app, pilot)
                assert not isinstance(app.screen, StarDialog)
            count_training(app.ws)
            await _back_home(app, pilot)
            assert isinstance(app.screen, StarDialog)
            await pilot.click("#never")
            await pilot.pause(0.2)
            for _ in range(ASK_AGAIN_AFTER + 1):
                count_training(app.ws)
            await _back_home(app, pilot)
            assert not isinstance(app.screen, StarDialog) and app.opened == []

    asyncio.run(scenario())


def test_not_during_the_tour_and_star_key(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.3)
            await pilot.press("f2")
            await pilot.pause(0.2)
            count_training(app.ws)
            await _back_home(app, pilot)
            assert not isinstance(app.screen, StarDialog)
            app.tour.end()
            await pilot.press("asterisk")
            await pilot.pause(0.2)
            assert app.opened == [feedback.REPO_URL]
            await _back_home(app, pilot)
            assert not isinstance(app.screen, StarDialog)  # they starred already

    asyncio.run(scenario())
