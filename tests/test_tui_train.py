"""The Train screen saves what the data taught the model, like the command line does."""
import asyncio

from aiforge import EXAMPLES_DIR
from aiforge.spec import Workspace, from_template
from aiforge.tui.app import AIForgeApp
from aiforge.tui.train import TrainScreen


def test_train_screen_saves_class_names(tmp_path):
    # This template already has 2 outputs, so the 2-class data is no "mismatch" and auto-fit
    # never runs - the class names must still be saved, or predictions say "class 0 / class 1".
    ws = Workspace(tmp_path)
    ws.save_spec(from_template("Bag of words (Embedding + average, fastest)", "reviews"))

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            preset = {"source": "csv_text", "path": str(EXAMPLES_DIR / "movie_reviews.csv")}
            await app.push_screen(TrainScreen("reviews", finetune=False, preset=preset))
            screen = app.screen
            for _ in range(200):
                await pilot.pause(0.05)
                if screen.info:
                    break
            assert screen.info, "data didn't load"
            screen.query_one("#epochs").value = "1"
            screen.query_one("#device").value = "cpu"
            screen.action_start()
            for _ in range(400):
                await pilot.pause(0.05)
                if ws.load_history("reviews"):
                    break

    asyncio.run(scenario())
    assert ws.load_history("reviews"), "training didn't finish"
    assert ws.load_spec("reviews").class_names == ["neg", "pos"]
