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


def test_train_screen_resumes_an_unfinished_run(tmp_path):
    from aiforge import pipeline
    from aiforge.trainer import TrainConfig

    ws = Workspace(tmp_path)
    ws.save_spec(from_template("MLP classifier (tabular)", "spiral"))
    spec = ws.load_spec("spiral")
    data = pipeline.load_data(ws, spec, {"source": "synthetic", "kind": "spirals"})
    box = []
    pipeline.train_model(ws, "spiral", data, TrainConfig(epochs=4, lr=0.005, max_samples=500, device="cpu"),
                         on_log=lambda m: None, on_trainer=box.append,
                         on_epoch=lambda r: box[0].stop() if r["epoch"] == 2 else None)

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await app.push_screen(TrainScreen("spiral", finetune=False))
            screen = app.screen
            await pilot.pause()
            button = screen.query_one("#resume-run")
            assert button.display and "2/4" in str(button.label)
            screen.query_one("#device").value = "cpu"
            button.press()
            for _ in range(400):
                await pilot.pause(0.05)
                if pipeline.interrupted_run(ws, "spiral") is None and not screen.trainer:
                    break
            assert screen.query_one("#max_samples").value == "500"  # the run's own settings
            assert len(screen.loss_hist) == 4  # earlier epochs redrawn, then the rest
            assert not button.display

            # A model without an unfinished run shows no Resume button.
            ws.save_spec(from_template("MLP classifier (tabular)", "other"))
            await app.push_screen(TrainScreen("other", finetune=False))
            await pilot.pause()
            assert not app.screen.query_one("#resume-run").display

    asyncio.run(scenario())
    hist = ws.load_history("spiral")
    assert len(hist) == 1 and hist[0]["resumed_at"] == 2 and len(hist[0]["epochs"]) == 4


def test_train_screen_loads_a_huggingface_dataset(tmp_path, hub):
    ws = Workspace(tmp_path)
    ws.save_spec(from_template("MLP classifier (tabular)", "iris"))

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await app.push_screen(TrainScreen("iris", finetune=False))
            screen = app.screen
            screen.query_one("#source").value = "hf"
            await pilot.pause()
            assert screen.query_one("#grp-hf").display and not screen.query_one("#grp-path").display
            screen.query_one("#hf_name").value = "iris"
            screen.query_one("#hf_target").value = "species"
            screen.query_one("#load").press()
            for _ in range(200):
                await pilot.pause(0.05)
                if screen.info:
                    break
            assert screen.info.class_names == ["setosa", "virginica"]
            assert screen.data_cfg["name"] == "iris" and screen.data_cfg["split"] == "train"

    asyncio.run(scenario())
