"""The goal wizard turns plain words and a data file into a model that is ready to train."""
import asyncio

import pytest
from PIL import Image

from aiforge import EXAMPLES_DIR
from aiforge.data import load
from aiforge.spec import Workspace
from aiforge.trainer import data_mismatch, fit_spec_to_data
from aiforge.wizard import PlanError, plan, read_goal


def _ready(p, tmp_path):
    """The planned model loads its data and, after auto-fit, has nothing left to fix."""
    spec = p.spec()
    _, _, info = load(p.data_cfg, tmp_path, spec)
    if data_mismatch(spec, info):
        fit_spec_to_data(spec, info)
    assert not data_mismatch(spec, info)
    assert not spec.validate()
    return spec, info


def test_a_number_to_predict_is_regression(tmp_path):
    p = plan("predict house prices", str(EXAMPLES_DIR / "house_prices.csv"))
    assert (p.template, p.task, p.label) == ("MLP regressor (tabular)", "regression", "price")
    assert p.name == "house-prices" and p.train["early_stop"] > 0
    assert _ready(p, tmp_path)[1].task == "regression"


def test_the_column_before_from_is_the_label():
    p = plan("predict price from rooms and area", str(EXAMPLES_DIR / "house_prices.csv"))
    assert p.label == "price"
    p = plan("estimate the area from the price", str(EXAMPLES_DIR / "house_prices.csv"))
    assert p.label == "area"


def test_a_label_column_by_name_and_a_chosen_one(tmp_path):
    p = plan("tell big fruit from small fruit", str(EXAMPLES_DIR / "fruit_size.csv"))
    assert (p.task, p.label) == ("classification", "label")
    _, info = _ready(p, tmp_path)
    assert info.class_names == ["big", "small"]
    p = plan("tell big fruit from small fruit", str(EXAMPLES_DIR / "fruit_size.csv"), label="color")
    assert p.label == "color" and "you chose it" in p.summary()


def test_a_table_of_texts_is_text_classification(tmp_path):
    p = plan("is a movie review positive or negative", str(EXAMPLES_DIR / "movie_reviews.csv"))
    assert p.data_cfg["source"] == "csv_text" and p.data_cfg["text_column"] == "text"
    assert p.label == "sentiment"
    assert _ready(p, tmp_path)[1].class_names == ["neg", "pos"]


def test_an_image_folder_with_sub_folders_is_classification(tmp_path):
    p = plan("sort pictures by colour", str(EXAMPLES_DIR / "color_images"))
    assert p.task == "classification" and p.template.startswith("CNN")  # small images: no download
    assert _ready(p, tmp_path)[1].class_names == ["blue", "red"]


def test_big_photos_start_from_a_pretrained_network(tmp_path):
    for cls in ("cats", "dogs"):
        (tmp_path / "pets" / cls).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (300, 200)).save(tmp_path / "pets" / cls / f"{i}.jpg")
    p = plan("sort photos of cats and dogs", str(tmp_path / "pets"))
    assert "Transfer learning" in p.template and p.train["lr"] < 0.001
    assert any("only 3 images" in w for w in p.warnings)


def test_a_folder_without_classes_finds_groups(tmp_path):
    for i in range(6):
        Image.new("RGB", (16, 16), (i * 40, 0, 0)).save(tmp_path / f"{i}.png")
    p = plan("sort my photos", str(tmp_path))
    assert p.task == "clustering" and any("sub-folder" in w for w in p.warnings)


def test_groups_and_oddities_need_no_labels(tmp_path):
    p = plan("find 3 groups of fruit", str(EXAMPLES_DIR / "fruit_size.csv"))
    assert p.task == "clustering" and p.groups == 3 and "(3 groups)" in p.summary()
    spec, info = _ready(p, tmp_path)
    assert spec.layers[0].config["k"] == 3 and info.class_names  # the label is only kept aside
    p = plan("spot unusual fruit", str(EXAMPLES_DIR / "fruit_size.csv"))
    assert p.task == "autoencoder"
    _ready(p, tmp_path)


def test_a_text_file_teaches_writing(tmp_path):
    p = plan("write nursery rhymes", str(EXAMPLES_DIR / "nursery_rhymes.txt"))
    assert p.task == "language_model" and any("short text" in w for w in p.warnings)
    _ready(p, tmp_path)


def test_goals_without_a_file_use_built_in_data():
    assert plan("recognise handwritten digits").data_cfg == {"source": "mnist"}
    assert plan("classify pictures of clothes").data_cfg == {"source": "fashion_mnist"}
    agent = plan("balance a pole on a cart")
    assert agent.task == "reinforcement" and agent.data_cfg["env"] == "cartpole"
    assert plan("write poems").task == "language_model"


def test_unclear_requests_say_what_to_add(tmp_path):
    with pytest.raises(PlanError, match="choose your data"):
        plan("sort my things")
    with pytest.raises(PlanError, match="Describe"):
        plan("")
    with pytest.raises(PlanError, match="Can't find"):
        plan("predict prices", str(tmp_path / "missing.csv"))


def test_without_data_it_suggests_famous_datasets_that_fit():
    with pytest.raises(PlanError) as e:
        plan("predict house prices")
    assert e.value.suggestions[0] == "california_housing" and "California housing" in str(e.value)
    with pytest.raises(PlanError) as e:
        plan("detect spam messages")
    assert e.value.suggestions == ["sms_spam"]
    with pytest.raises(PlanError) as e:
        plan("sort photos of cats and dogs")
    assert set(e.value.suggestions) == {"cifar10", "oxford_pets"}  # not the clothes


def test_a_built_in_dataset_by_name():
    p = plan("tell the objects apart", "cifar10")
    assert p.data_cfg == {"source": "cifar10"} and p.template.startswith("CNN")


def test_an_id_column_gets_a_warning(tmp_path):
    rows = "\n".join(f"{i},{i % 7},{i % 3}" for i in range(100))
    (tmp_path / "t.csv").write_text("customer_id,visits,plan\n" + rows)
    p = plan("predict the plan", str(tmp_path / "t.csv"))
    assert p.label == "plan" and any("customer_id" in w for w in p.warnings)


def test_reading_the_goal():
    assert read_goal("How much will this house sell for?").task == "regression"
    assert read_goal("predict whether a customer leaves").task == "classification"
    assert read_goal("predict churn").task is None  # the label column decides
    assert read_goal("split shoppers into five groups").groups == 5


def test_the_wizard_creates_a_model_and_opens_it_ready_to_train(tmp_path):
    from aiforge.tui.app import AIForgeApp
    from aiforge.tui.train import TrainScreen
    from aiforge.tui.wizard import GoalDialog

    ws = Workspace(tmp_path)

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.press("n")
            await pilot.pause()
            await pilot.click("#wizard")
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, GoalDialog)
            dialog.query_one("#goal").value = "predict house prices"
            dialog.query_one("#data").value = str(EXAMPLES_DIR / "house_prices.csv")
            await pilot.click("#make-plan")
            for _ in range(100):
                await pilot.pause(0.05)
                if dialog.plan:
                    break
            assert dialog.plan and dialog.query_one("#label").value == "price"
            await pilot.click("#ok")
            for _ in range(200):
                await pilot.pause(0.05)
                screen = app.screen
                if isinstance(screen, TrainScreen) and screen.info:
                    break
            assert app.screen is not dialog, dialog.query_one("#error").render()
            assert isinstance(screen, TrainScreen) and screen.info, "data didn't load"
            assert screen.query_one("#epochs").value == "60"
            assert screen.query_one("#early_stop").value == "8"
            assert not data_mismatch(screen.spec, screen.info)  # fitted without asking

    asyncio.run(scenario())
    spec = ws.load_spec("house-prices")
    assert spec.task == "regression" and spec.data["dataset_cfg"]["target"] == "price"


def test_a_famous_dataset_from_the_hub_brings_its_label_column(tmp_path):
    from aiforge.tui.app import AIForgeApp
    from aiforge.tui.wizard import GoalDialog

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            dialog = GoalDialog(app.ws, goal="guess the size of a home")
            await app.push_screen(dialog)
            # what the hub returns for a downloaded table (the label isn't the last column here)
            dialog._use_dataset({"source": "csv", "path": str(EXAMPLES_DIR / "house_prices.csv"),
                                 "target": "area", "task": "regression", "hub": "houses"})
            for _ in range(100):
                await pilot.pause(0.05)
                if dialog.plan:
                    break
            assert dialog.plan.label == "area"
            assert dialog.query_one("#data").value.endswith("house_prices.csv")

    asyncio.run(scenario())
