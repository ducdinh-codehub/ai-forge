"""Serve a trained model over HTTP: a JSON API plus a small demo page.

    aiforge serve my-model            # then open http://127.0.0.1:8000

Needs the optional extra: ``pip install "aiforge-studio[serve]"`` (FastAPI + uvicorn).
"""
# No "from __future__ import annotations": FastAPI reads the request models from the real annotations.

import html
import json
import tempfile
import threading
from pathlib import Path

from .data import IMAGE_EXT
from .predict import input_hint, predict_one
from .spec import Workspace

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_BATCH = 1000
MAX_GENERATE = 2000
INSTALL_HINT = ('Serving needs FastAPI and uvicorn: pip install "aiforge-studio[serve]" '
                "(with pipx: pipx inject aiforge-studio fastapi uvicorn python-multipart)")


def missing_dependencies() -> str | None:
    try:
        import fastapi  # noqa: F401
        import multipart  # noqa: F401 - FastAPI needs it for image uploads
        import uvicorn  # noqa: F401
    except ImportError:
        return INSTALL_HINT
    return None


def input_kind(spec) -> str:
    """How a request carries one input: 'image' (an uploaded file), 'prompt' or 'text'."""
    kind = spec.data.get("preprocess", {}).get("type")
    if kind == "image" or (kind is None and len(spec.input_shape) == 3):
        return "image"
    if spec.task == "language_model":
        return "prompt"
    return "text"


def create_app(ws: Workspace, name: str):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field

    spec, model, _ = ws.load_model(name)
    model.eval()
    kind = input_kind(spec)
    lock = threading.Lock()  # one forward pass at a time: the model is shared by every request

    class PredictIn(BaseModel):
        input: str | None = Field(None, description="One input: comma-separated values or a text")
        inputs: list[str] | None = Field(None, description=f"Several inputs (at most {MAX_BATCH})")
        top_k: int = Field(5, ge=1, le=100)

    class GenerateIn(BaseModel):
        prompt: str = ""
        length: int = Field(200, ge=1, le=MAX_GENERATE)
        temperature: float = Field(0.8, gt=0, le=5)
        seed: int | None = None

    def run(raw: str, top_k: int) -> list[dict]:
        try:
            with lock:
                res = predict_one(spec, model, raw, top_k=top_k)
        except (ValueError, KeyError) as e:
            raise HTTPException(422, str(e.args[0] if e.args else e)) from None
        return [{"label": label, "score": score} for label, score in res]

    app = FastAPI(title=f"AIForge · {name}", version="1",
                  description=f"A {spec.task} model trained with AIForge. Input: {input_hint(spec)}")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page():
        return demo_page(name, spec, kind)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        return {"model": name, "task": spec.task, "input": kind, "input_hint": input_hint(spec),
                "input_shape": spec.input_shape, "classes": spec.class_names,
                "params": spec.total_params(),
                "endpoints": (["/generate", "/predict"] if kind == "prompt" else
                              ["/predict/image"] if kind == "image" else ["/predict"])}

    @app.post("/predict")
    def predict(body: PredictIn):
        if kind == "image":
            # Never take a path from the client: that would let callers read files on this machine.
            raise HTTPException(422, "This is an image model - upload the image to /predict/image")
        if (body.input is None) == (body.inputs is None):
            raise HTTPException(422, "Send either 'input' (one) or 'inputs' (a list)")
        if body.inputs is not None:
            if len(body.inputs) > MAX_BATCH:
                raise HTTPException(422, f"At most {MAX_BATCH} inputs per request")
            return {"predictions": [run(x, body.top_k) for x in body.inputs]}
        return {"predictions": run(body.input, body.top_k)}

    @app.post("/predict/image")
    def predict_image(file: UploadFile = File(...), top_k: int = 5):
        if kind != "image":
            raise HTTPException(422, "This model doesn't take images - use /predict")
        data = file.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"Images can be at most {MAX_IMAGE_BYTES // 2 ** 20} MB")
        suffix = Path(file.filename or "").suffix.lower()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix if suffix in IMAGE_EXT else ''}"
            path.write_bytes(data)
            try:
                return {"predictions": run(str(path), max(1, min(top_k, 100)))}
            except HTTPException:
                raise
            except Exception:  # noqa: BLE001 - PIL raises many types for broken files
                raise HTTPException(422, "That file isn't an image AIForge can read") from None

    if kind == "prompt":
        from .textgen import generate

        @app.post("/generate")
        def write(body: GenerateIn):
            with lock:
                text = generate(model, spec.data["preprocess"], body.prompt, length=body.length,
                                temperature=body.temperature, seed=body.seed)
            return {"prompt": body.prompt, "text": text}

    return app


def port_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as sock:
        # Same as uvicorn: a port a stopped server just left (TIME_WAIT) counts as free.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def check_servable(ws: Workspace, name: str) -> None:
    if not ws.has_weights(name):
        raise ValueError(f"'{name}' isn't trained yet - train it before serving it")
    spec = ws.load_spec(name)
    if spec.task == "language_model" and spec.data.get("preprocess", {}).get("type") != "corpus":
        raise ValueError("This language model has no vocabulary saved - train it in AIForge first")


class Server:
    """Runs uvicorn in a background thread (for the TUI); ``stop()`` shuts it down."""

    def __init__(self, ws: Workspace, name: str, host: str = "127.0.0.1", port: int = 8000):
        import logging

        import uvicorn

        # Messages printed from a background thread would draw over the TUI.
        for logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(logger).addHandler(logging.NullHandler())
            logging.getLogger(logger).propagate = False
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
        config = uvicorn.Config(create_app(ws, name), host=host, port=port, log_config=None,
                                log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout: float = 10) -> None:
        cfg = self._server.config
        if not port_free(cfg.host, cfg.port):
            raise OSError(f"Port {cfg.port} is already in use - stop the other server or pick another port")
        self._thread.start()
        for _ in range(int(timeout / 0.05)):
            if self._server.started:
                return
            if not self._thread.is_alive():  # e.g. the port is taken
                break
            self._thread.join(0.05)
        raise OSError(f"The server didn't start - is port {self._server.config.port} already in use?")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(5)


def serve_forever(ws: Workspace, name: str, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(ws, name), host=host, port=port, log_level="info")


# -- demo page -------------------------------------------------------------------------------


def demo_page(name: str, spec, kind: str) -> str:
    hint = html.escape(input_hint(spec))
    if kind == "image":
        form = '<input type="file" id="file" accept="image/*" required><img id="preview" alt="">'
    elif kind == "prompt":
        form = (f'<textarea id="text" rows="4" placeholder="{hint}"></textarea>'
                '<label>Length <input type="number" id="length" value="200" min="1" max="2000"></label>'
                '<label>Temperature <input type="number" id="temp" value="0.8" step="0.1" min="0.1" max="5">'
                '</label>')
    else:
        rows = 4 if spec.data.get("preprocess", {}).get("type") == "text" else 1
        form = f'<textarea id="text" rows="{rows}" placeholder="{hint}"></textarea>'
    meta = {"task": spec.task, "kind": kind}
    return (PAGE.replace("{{name}}", html.escape(name)).replace("{{hint}}", hint)
            .replace("{{task}}", html.escape(spec.task.replace("_", " ")))
            .replace("{{form}}", form).replace("{{meta}}", json.dumps(meta))
            .replace("{{button}}", "Write" if kind == "prompt" else "Predict"))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{name}} · AIForge</title>
<style>
:root { --bg:#f7f7f8; --card:#fff; --fg:#1b1b1f; --dim:#6b6b76; --line:#e2e2e7; --acc:#7b2cbf; --bar:#c9a7ef; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141418; --card:#1d1d23; --fg:#ececf1; --dim:#9a9aa6; --line:#2e2e37; --acc:#b583f0; --bar:#5b3a85; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
main { max-width:640px; margin:0 auto; padding:32px 16px; }
h1 { margin:0; font-size:1.5rem; } .dim { color:var(--dim); font-size:.9rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; margin-top:20px; }
textarea, input[type=number] { width:100%; padding:10px; border:1px solid var(--line); border-radius:8px;
  background:var(--bg); color:var(--fg); font:inherit; }
input[type=number] { width:7em; margin-left:8px; } label { display:inline-block; margin:12px 16px 0 0; }
input[type=file] { color:var(--fg); }
#preview { display:block; max-width:100%; max-height:220px; margin-top:12px; border-radius:8px; }
button { margin-top:16px; padding:10px 22px; border:0; border-radius:8px; background:var(--acc); color:#fff;
  font:inherit; font-weight:600; cursor:pointer; } button:disabled { opacity:.5; cursor:wait; }
.row { display:grid; grid-template-columns:minmax(0,9em) 1fr 4.5em; gap:10px; align-items:center; margin:6px 0; }
.row span:first-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bar { height:12px; background:var(--bar); border-radius:6px; } .row b { text-align:right; }
.big { font-size:1.8rem; font-weight:700; } .err { color:#d64545; }
pre { white-space:pre-wrap; margin:0; font:15px/1.5 ui-monospace, monospace; } pre b { color:var(--acc); }
a { color:var(--acc); }
</style></head>
<body><main>
<h1>{{name}}</h1>
<div class="dim">A {{task}} model trained with AIForge · <a href="/docs">API docs</a></div>
<form class="card" id="form">
  <div class="dim" style="margin-bottom:8px">{{hint}}</div>
  {{form}}
  <button id="go">{{button}}</button>
</form>
<div class="card" id="out" hidden></div>
</main>
<script>
const meta = {{meta}};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
if ($("file")) $("file").onchange = e => {
  const f = e.target.files[0]; if (f) $("preview").src = URL.createObjectURL(f);
};
function show(html) { $("out").hidden = false; $("out").innerHTML = html; }
$("form").onsubmit = async e => {
  e.preventDefault(); $("go").disabled = true;
  try {
    let res;
    if (meta.kind === "image") {
      const fd = new FormData(); fd.append("file", $("file").files[0]);
      res = await fetch("/predict/image", {method:"POST", body:fd});
    } else if (meta.kind === "prompt") {
      res = await fetch("/generate", {method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({prompt:$("text").value, length:+$("length").value, temperature:+$("temp").value})});
    } else {
      res = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({input:$("text").value})});
    }
    const data = await res.json();
    if (!res.ok) {
      const d = data.detail; show(`<div class="err">${esc(Array.isArray(d) ? d.map(x => x.msg).join("; ") : d)}</div>`);
    } else if (meta.kind === "prompt") {
      show(`<pre><b>${esc(data.prompt)}</b>${esc(data.text)}</pre>`);
    } else if (meta.task === "regression") {
      show(data.predictions.map(p => `<div class="dim">${esc(p.label)}</div><div class="big">${
        Number(p.score.toPrecision(6))}</div>`).join(""));
    } else {
      show(data.predictions.map(p => `<div class="row"><span>${esc(p.label)}</span>
        <div class="bar" style="width:${Math.max(1, p.score * 100)}%"></div><b>${(p.score * 100).toFixed(1)}%</b></div>`
      ).join(""));
    }
  } catch (err) { show(`<div class="err">${esc(err)}</div>`); }
  $("go").disabled = false;
};
</script>
</body></html>
"""
