"""The guided first run: offered once, walks the quick start, each step ends when it is done."""
import asyncio

from textual.widgets import Input

from aiforge.spec import Workspace, from_template
from aiforge.tui.app import AIForgeApp
from aiforge.tui.tour import STEPS


async def _until(pilot, cond, timeout=60):
    for _ in range(int(timeout * 20)):
        if cond():
            return
        await pilot.pause(0.05)
    raise AssertionError("timed out")


def _coach_says(app, text):
    try:
        return text in str(app.screen.query_one("#coach-text").render())
    except Exception:  # noqa: BLE001 - not mounted yet
        return False


def _outlined(app):
    return [w.id for w in app.screen.query(".tour-target")]


def test_tour_walks_the_quick_start(tmp_path):
    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.3)
            assert _coach_says(app, "New here?")  # offered on the first launch
            await pilot.click("#tour-start")
            await _until(pilot, lambda: _coach_says(app, "Step 1 of 5"))
            assert _outlined(app) == ["new"]

            await pilot.press("n")
            await _until(pilot, lambda: _coach_says(app, "MLP classifier"))
            assert app.screen.query_one("#name", Input).value == "spiral-classifier"  # prefilled
            assert _outlined(app) == ["ok"]
            app.screen.submit()
            await _until(pilot, lambda: _coach_says(app, "Step 2 of 5"))
            await pilot.press("ctrl+s")
            await pilot.press("escape")
            await _until(pilot, lambda: _coach_says(app, "Step 3 of 5"))
            assert _outlined(app) == ["train"]

            await pilot.press("t")
            await _until(pilot, lambda: _coach_says(app, "Load data"))
            s = app.screen
            assert s.query_one("#kind").value == "spirals" and s.query_one("#epochs").value == "40"
            s.query_one("#device").value = "cpu"
            s.query_one("#epochs").value = "5"
            s.query_one("#load").press()
            await _until(pilot, lambda: _coach_says(app, "Auto-fit"))
            assert _outlined(app) == ["autofit"]
            s.query_one("#autofit").press()
            await _until(pilot, lambda: _coach_says(app, "▶ Start"))
            s.query_one("#start").press()
            await _until(pilot, lambda: _coach_says(app, "Trained!"))
            await pilot.press("escape")
            await _until(pilot, lambda: _coach_says(app, "Step 4 of 5"))
            await pilot.press("p")
            await _until(pilot, lambda: _coach_says(app, "0.1, 0.45"))
            app.screen.query_one("#raw", Input).value = "0.1, 0.45"
            await pilot.press("enter")
            await _until(pilot, lambda: _coach_says(app, "Step 5 of 5"))
            assert _outlined(app) == ["explain"]
            assert str(app.screen.query_one("#tour-end").label) == "Finish"
            await pilot.click("#tour-end")
            await _until(pilot, lambda: app.tour is None)
            await _until(pilot, lambda: not app.screen.query(".coach"))  # removed a moment later
            assert not _outlined(app)

    asyncio.run(scenario())


def test_offered_once_and_not_to_people_with_models(tmp_path):
    async def launch(ws_dir):
        app = AIForgeApp(ws_dir)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.3)
            offered = _coach_says(app, "New here?")
            if offered:
                await pilot.click("#tour-no")
                await _until(pilot, lambda: not app.screen.query(".coach"))
            return offered

    assert asyncio.run(launch(tmp_path / "a")) is True
    assert asyncio.run(launch(tmp_path / "a")) is False  # once is enough
    Workspace(tmp_path / "b").save_spec(from_template("MLP classifier (tabular)", "m"))
    assert asyncio.run(launch(tmp_path / "b")) is False


def test_f2_restarts_and_skip_ends_it(tmp_path):
    Workspace(tmp_path).save_spec(from_template("MLP classifier (tabular)", "m"))

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.3)
            await pilot.press("f2")
            await _until(pilot, lambda: _coach_says(app, "Step 1 of 5"))
            await pilot.click("#tour-end")
            await _until(pilot, lambda: app.tour is None)
            await _until(pilot, lambda: not app.screen.query(".coach"))  # removed a moment later
            assert not _outlined(app)

    asyncio.run(scenario())


def test_every_step_has_a_screen_and_text():
    assert STEPS[-1].done(None) is False  # the last step waits for Finish
    assert all(s.screen and s.text for s in STEPS)


# -- the agent tour (reinforcement learning) ------------------------------------------------------


def test_agent_tour_walks_a_first_agent(tmp_path):
    from aiforge.tui.tour import AGENT_STEPS, start

    ws = Workspace(tmp_path)
    ws.set_setting("tour_offered", True)
    ws.set_setting("star_answered", True)

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause(0.3)
            start(app, AGENT_STEPS)
            await _until(pilot, lambda: _coach_says(app, "Step 1 of 6"))
            await pilot.press("n")
            await _until(pilot, lambda: _coach_says(app, "Reinforcement"))
            assert _outlined(app) == ["style-reinforcement"]
            assert app.screen.query_one("#name", Input).value == "grid-explorer"
            await _until(pilot, lambda: app.screen.query_one("#style-reinforcement").region.width > 0)
            await pilot.click("#style-reinforcement")
            await _until(pilot, lambda: _coach_says(app, "GridWorld agent"))
            app.screen.submit()
            await _until(pilot, lambda: _coach_says(app, "Step 2 of 6"))
            await pilot.press("ctrl+s")
            await pilot.press("escape")
            await _until(pilot, lambda: _coach_says(app, "Step 3 of 6"))
            await pilot.press("t")
            await _until(pilot, lambda: _coach_says(app, "Set up"))
            s = app.screen
            assert s.query_one("#epochs").value == "300" and _outlined(app) == ["load"]
            s.query_one("#epochs").value = "20"
            s.query_one("#load").press()
            await _until(pilot, lambda: _coach_says(app, "Step 4 of 6"))  # no auto-fit needed: skipped
            assert _outlined(app) == ["start"]
            s.query_one("#start").press()
            await _until(pilot, lambda: _coach_says(app, "Trained!"), timeout=120)
            await pilot.press("escape")
            await _until(pilot, lambda: _coach_says(app, "Step 5 of 6"))
            await pilot.press("p")
            await _until(pilot, lambda: _coach_says(app, "▶ Play"))
            use = app.screen
            while not use.done:  # step through one episode
                use.query_one("#step").press()
                await pilot.pause(0.02)
            await _until(pilot, lambda: _coach_says(app, "rewards alone"))
            await pilot.press("escape")
            await _until(pilot, lambda: _coach_says(app, "Step 6 of 6"))
            await pilot.press("v")
            await _until(pilot, lambda: _coach_says(app, "100 test episodes"))
            app.screen.query_one("#eval-episodes").value = "10"
            app.screen.query_one("#eval-go").press()
            await _until(pilot, lambda: _coach_says(app, "🎉 Done!"), timeout=60)
            assert str(app.screen.query_one("#tour-end").label) == "Finish"
            await pilot.click("#tour-end")
            await _until(pilot, lambda: app.tour is None)

    asyncio.run(scenario())
    assert ws.load_history("grid-explorer")


def test_agent_tour_is_offered_on_a_new_agents_train_screen(tmp_path):
    from aiforge.tui.train import TrainScreen

    ws = Workspace(tmp_path)
    ws.set_setting("tour_offered", True)
    ws.save_spec(from_template("CartPole agent (DQN)", "pole"))

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause(0.3)
            await app.push_screen(TrainScreen("pole", finetune=False))
            await _until(pilot, lambda: _coach_says(app, "First agent?"))
            # The bar was just added: click only once it is laid out (slow CI machines lag behind).
            await _until(pilot, lambda: app.screen.query_one("#agent-tour-start").region.width > 0)
            await pilot.click("#agent-tour-start")
            await _until(pilot, lambda: _coach_says(app, "CartPole"))  # starts right here, in its own world
            assert app.tour.steps[app.tour.i].screen == "TrainScreen"
            assert app.screen.query_one("#env").value == "cartpole"

    asyncio.run(scenario())
    assert ws.get_setting("agent_tour_offered") is True  # offered once only


def test_every_agent_step_has_a_screen_and_text():
    from aiforge.tui.tour import AGENT_STEPS

    assert AGENT_STEPS[-1].done(None) is False
    texts = [s.text(None) if callable(s.text) else s.text for s in AGENT_STEPS]
    assert all(s.screen for s in AGENT_STEPS) and all(texts)
    assert sum("of 6" in t for t in texts) == 6
