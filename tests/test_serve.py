"""The web API: one model of each input kind, plus the safety rules."""
import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aiforge import EXAMPLES_DIR  # noqa: E402
from aiforge.cli import main  # noqa: E402
from aiforge.serve import create_app  # noqa: E402
from aiforge.spec import Workspace  # noqa: E402


def trained(ws_dir, name, template, data, *extra):
    run = lambda *a: main(["-w", str(ws_dir), *map(str, a)])  # noqa: E731
    assert run("new", name, "-t", template) == 0
    assert run("train", name, "-d", data, "-e", 2, "--device", "cpu", *extra) == 0
    return TestClient(create_app(Workspace(ws_dir), name))


@pytest.fixture(scope="module")
def ws_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("ws")


@pytest.fixture(scope="module")
def spiral(ws_dir):
    return trained(ws_dir, "spiral", "mlp classifier", "spirals")


def test_page_info_health(spiral):
    assert spiral.get("/health").json() == {"status": "ok"}
    info = spiral.get("/info").json()
    assert info["classes"] == ["arm-0", "arm-1", "arm-2"] and info["input"] == "text"
    page = spiral.get("/")
    assert page.status_code == 200 and "spiral" in page.text and "/predict" in page.text
    assert spiral.get("/docs").status_code == 200


def test_predict_one_and_many(spiral):
    one = spiral.post("/predict", json={"input": "0.1, 0.45", "top_k": 2}).json()["predictions"]
    assert len(one) == 2 and one[0]["label"].startswith("arm-") and 0 <= one[0]["score"] <= 1
    many = spiral.post("/predict", json={"inputs": ["0.1, 0.45", "-0.3, 0.2"]}).json()["predictions"]
    assert len(many) == 2 and len(many[0]) == 3


@pytest.mark.parametrize("body", [{"input": "1, 2, 3"}, {}, {"input": "1,2", "inputs": ["1,2"]},
                                  {"inputs": ["0,0"] * 1001}])
def test_bad_requests_are_422(spiral, body):
    r = spiral.post("/predict", json=body)
    assert r.status_code == 422 and r.json()["detail"]


def test_image_model_takes_uploads_only(ws_dir):
    client = trained(ws_dir, "colours", "small cnn", EXAMPLES_DIR / "color_images")
    img = next((EXAMPLES_DIR / "color_images" / "blue").glob("*.png"))
    r = client.post("/predict/image", files={"file": ("x.png", img.read_bytes(), "image/png")})
    assert r.status_code == 200 and r.json()["predictions"][0]["label"] in client.get("/info").json()["classes"]
    # A path in JSON would make the server open files on its own disk - always refused.
    assert client.post("/predict", json={"input": str(img)}).status_code == 422
    junk = client.post("/predict/image", files={"file": ("x.png", io.BytesIO(b"not an image"), "image/png")})
    assert junk.status_code == 422


def test_language_model_writes(ws_dir):
    client = trained(ws_dir, "gpt", "tiny gpt", EXAMPLES_DIR / "nursery_rhymes.txt", "--max-samples", 200)
    r = client.post("/generate", json={"prompt": "Mary", "length": 20, "seed": 1})
    assert r.status_code == 200 and r.json()["text"]
    assert client.post("/generate", json={"length": 999999}).status_code == 422
    assert client.post("/predict", json={"input": "Mar"}).json()["predictions"]  # next-token guesses


def test_untrained_model_is_refused(ws_dir):
    assert main(["-w", str(ws_dir), "new", "blank", "-t", "1"]) == 0
    assert main(["-w", str(ws_dir), "serve", "blank"]) == 1
