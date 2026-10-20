"""Regenerate the README screenshots in docs/images/.

Run from the project root:  .venv/bin/python docs/make_screenshots.py
It drives the app through a real session (create, build, train, fine-tune, use, export,
download a dataset - this step needs internet)
in a throw-away workspace, captures each screen as SVG and converts it to PNG with headless
Google Chrome (PNG, because PyPI can't display SVGs hosted on GitHub).
"""
import asyncio, shutil, sys, tempfile, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aiforge import EXAMPLES_DIR
from aiforge.tui.app import AIForgeApp
from textual.widgets import DataTable, Input, Select, SelectionList, OptionList

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(exist_ok=True)
SVG_DIR = Path(tempfile.mkdtemp(prefix="aiforge-svg-"))
CHROME = next((p for p in [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chrome") or "",
] if p and Path(p).exists()), None)


def svg_to_png(svg: Path, png: Path) -> None:
    """Render an SVG at its natural size with headless Chrome."""
    import math, re, subprocess

    w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg.read_text()).groups())
    w, h = math.ceil(w), math.ceil(h)
    page = svg.with_suffix(".html")
    page.write_text(f'<html><body style="margin:0;background:#000"><img src="{svg.as_uri()}" '
                    f'style="display:block;width:{w}px;height:{h}px"></body></html>')
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=1", f"--window-size={w},{h}",
                    f"--screenshot={png}", page.as_uri()], check=True, capture_output=True)
WS = Path(tempfile.mkdtemp(prefix="aiforge-shots-"))
SIZE = (150, 44)

async def wait_for(pilot, cond, timeout=120):
    for _ in range(int(timeout * 10)):
        if cond(): return
        await pilot.pause(0.1)
    raise TimeoutError

async def main():
    app = AIForgeApp(WS)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.3)
        app.sub_title = "workspace: ./workspace"; await pilot.pause(0.3)
        def shot(name, title):
            svg = SVG_DIR / f"{name}.svg"
            app.save_screenshot(str(svg))
            svg.write_text(svg.read_text().replace(">AIForgeApp<", f">{title}<"))
            if CHROME:
                svg_to_png(svg, OUT / f"{name}.png")
            else:
                shutil.copy(svg, OUT / f"{name}.svg")
            print("saved", name)
        shot("01-home-empty", "AIForge – first launch")

        # New model dialog
        await pilot.press("n"); await pilot.pause()
        app.screen.query_one("#name", Input).value = "spiral-classifier"
        await pilot.pause(0.3)
        shot("02-new-model", "AIForge – create a new model")
        app.screen.submit(); await pilot.pause(0.6)

        # Builder: select a Linear layer so the settings panel shows
        b = app.screen; t = b.query_one("#stack", DataTable)
        t.focus(); t.move_cursor(row=4); await pilot.pause(0.5)
        pal = b.query_one("#palette", OptionList)
        pal.highlighted = next(i for i in range(pal.option_count) if pal.get_option_at_index(i).id == "Dropout")
        await pilot.pause(0.3)
        shot("03-builder", "AIForge – model builder")
        await pilot.press("ctrl+e"); await pilot.pause(0.6)
        shot("12-builder-explain", "AIForge – the Explain tab")
        await pilot.press("ctrl+e"); await pilot.pause(0.3)
        t.focus(); t.move_cursor(row=0); await pilot.pause(0.3)
        await pilot.press("question_mark"); await pilot.pause(0.8)
        shot("13-guide-overview", "AIForge – model guide")
        nav = app.screen.query_one("#guide-nav", OptionList)
        nav.highlighted = next(i for i in range(nav.option_count)
                               if nav.get_option_at_index(i).id == "type:Conv2d")
        await pilot.pause(0.8)
        shot("14-guide-encyclopedia", "AIForge – layer encyclopedia")
        await pilot.press("escape"); await pilot.pause(0.3)
        await pilot.press("ctrl+s"); await pilot.pause(); await pilot.press("escape"); await pilot.pause()

        # Builder error example (CNN template with a misplaced layer) -- separate model
        await pilot.press("n"); await pilot.pause()
        app.screen.query_one("#name", Input).value = "digit-reader"
        app.screen.query_one("#type", Select).value = "🖼 Computer vision (images)"; await pilot.pause(0.3)
        app.screen.query_one("#template", Select).value = "Small CNN (28×28 grayscale, e.g. MNIST)"
        app.screen.submit(); await pilot.pause(0.6)
        b = app.screen; t = b.query_one("#stack", DataTable)
        t.focus(); t.move_cursor(row=9); await pilot.pause(0.3)
        b.add_layer("Conv2d"); await pilot.pause(0.6)
        shot("04-builder-error", "AIForge – the builder catches mistakes")
        await pilot.press("ctrl+z"); await pilot.pause(0.3)
        await pilot.press("ctrl+s"); await pilot.pause(); await pilot.press("escape"); await pilot.pause()

        # Train the spiral model
        home = app.screen
        home.query_one("#models", DataTable).move_cursor(row=1); await pilot.pause()
        await pilot.press("t"); await pilot.pause(0.6)
        s = app.screen
        s.query_one("#source", Select).value = "csv"; await pilot.pause(0.3)
        s.query_one("#browse-data").press(); await pilot.pause(0.5)
        fb = app.screen
        fb.go(EXAMPLES_DIR); await pilot.pause(0.5)
        fb._set_selected(EXAMPLES_DIR / "fruit_size.csv")
        await pilot.pause(0.4)
        shot("15-file-browser", "AIForge – browsing for data")
        await pilot.press("escape"); await pilot.pause(0.3)
        s.query_one("#source", Select).value = "synthetic"; await pilot.pause(0.3)
        s.query_one("#epochs", Input).value = "40"
        s.query_one("#lr", Input).value = "0.005"
        s.query_one("#load").press()
        await wait_for(pilot, lambda: s.info is not None); await pilot.pause(0.3)
        shot("05-train-mismatch", "AIForge – data loaded, model needs auto-fit")
        await pilot.click("#autofit"); await pilot.pause(0.3)
        await pilot.click("#start")
        await wait_for(pilot, lambda: s.trainer is None and s.loss_hist); await pilot.pause(0.5)
        shot("06-train-done", "AIForge – training")
        s.query_one("#guide-btn").press(); await pilot.pause(0.6)
        shot("16-train-guide", "AIForge – guide panel while training")
        s.query_one("#guide-drawer").display = False; await pilot.pause(0.2)
        await pilot.press("escape"); await pilot.pause()

        # Fine-tune: load blobs, auto-fit, freeze body
        home.query_one("#models", DataTable).move_cursor(row=1); await pilot.pause()
        await pilot.press("f"); await pilot.pause(0.6)
        s = app.screen
        s.query_one("#kind", Select).value = "blobs"; await pilot.pause()
        s.query_one("#load").press()
        await wait_for(pilot, lambda: s.info is not None)
        await pilot.click("#autofit"); await pilot.pause(0.3)
        s.query_one("#freeze-body").press(); await pilot.pause(0.3)
        s.query_one("#train-config").scroll_end(animate=False); await pilot.pause(0.3)
        shot("07-finetune", "AIForge – fine-tuning")
        await pilot.press("escape"); await pilot.pause()

        # Predict
        home.query_one("#models", DataTable).move_cursor(row=1); await pilot.pause()
        await pilot.press("p"); await pilot.pause(0.6)
        s = app.screen
        s.query_one("#raw", Input).value = "0.1, 0.45"; await pilot.press("enter"); await pilot.pause(0.3)
        shot("08-predict", "AIForge – using a trained model")
        await pilot.press("escape"); await pilot.pause()

        # Export
        await pilot.press("x"); await pilot.pause(0.6)
        s = app.screen
        sl = s.query_one("#formats", SelectionList); sl.select("onnx"); sl.select("pytorch_py"); await pilot.pause()
        s.query_one("#out", Input).value = "./workspace/exports/spiral-classifier"
        await pilot.click("#go")
        await wait_for(pilot, lambda: any("Done" in str(l.text) for l in s.query_one("#export-log").lines))
        await pilot.pause(0.3)
        shot("09-export", "AIForge – exporting")
        await pilot.press("escape"); await pilot.pause()

        # Home with models
        home.query_one("#models", DataTable).move_cursor(row=1); await pilot.pause(0.5)
        shot("10-home", "AIForge – model library")

        # Dataset hub (needs internet): a downloaded dataset, then a download in progress
        await pilot.press("s"); await pilot.pause(0.5)
        hub = app.screen
        hub_list = hub.query_one("#hub-list", OptionList)

        def pick(key):
            hub_list.highlighted = next(i for i in range(hub_list.option_count)
                                        if hub_list.get_option_at_index(i).id == key)
        pick("iris"); await pilot.pause(0.3)
        hub._start_download()
        await wait_for(pilot, lambda: hub.downloading is None); await pilot.pause(0.5)
        md = hub.query_one("#hub-text")  # show the throw-away workspace as ./workspace
        md.update(md.source.replace(str(WS), "./workspace")); await pilot.pause(0.5)
        shot("17-datasets", "AIForge – download famous datasets")
        pick("imdb"); await pilot.pause(0.3)
        hub._start_download()
        await wait_for(pilot, lambda: "MB /" in str(hub.query_one("#hub-status").render()))
        await pilot.pause(3)
        shot("18-datasets-download", "AIForge – downloading a dataset")
        hub._cancel = True
        await wait_for(pilot, lambda: hub.downloading is None)
        await pilot.press("escape"); await pilot.pause(0.3)

        await pilot.press("i"); await pilot.pause(0.3)
        app.screen.query_one("#path", Input).value = "~/Downloads/my-model.aiforge.zip"; await pilot.pause()
        shot("11-import", "AIForge – import a model")

if not CHROME:
    print("Google Chrome not found - saving SVG screenshots instead of PNG.")
asyncio.run(main())
shutil.rmtree(WS, ignore_errors=True)
shutil.rmtree(SVG_DIR, ignore_errors=True)
