"""Regenerate the README demo animation, docs/images/demo.gif.

Run from the project root:  .venv/bin/python docs/make_demo_gif.py
It drives the app through the 5-minute quick start (create, build, train, use) in a throw-away
workspace, captures frames as SVG, renders them to PNG with headless Google Chrome and joins
them into a GIF with Pillow.
"""
import asyncio, math, re, shutil, subprocess, sys, tempfile, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageDraw, ImageFont
from aiforge.tui.app import AIForgeApp
from textual.widgets import Input

OUT = Path(__file__).resolve().parent / "images" / "demo.gif"
FRAMES = Path(tempfile.mkdtemp(prefix="aiforge-gif-"))
WS = Path(tempfile.mkdtemp(prefix="aiforge-demo-"))
SIZE = (130, 38)
WIDTH = 1040  # px of the final GIF: readable on GitHub, a few MB
CHROME = next((p for p in [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chrome") or "",
] if p and Path(p).exists()), None)


def svg_to_png(svg: Path, png: Path) -> None:
    w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg.read_text()).groups())
    w, h = math.ceil(w), math.ceil(h)
    page = svg.with_suffix(".html")
    page.write_text(f'<html><body style="margin:0;background:#000"><img src="{svg.as_uri()}" '
                    f'style="display:block;width:{w}px;height:{h}px"></body></html>')
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=1", f"--window-size={w},{h}",
                    f"--screenshot={png}", page.as_uri()], check=True, capture_output=True)


async def wait_for(pilot, cond, timeout=120):
    for _ in range(int(timeout * 20)):
        if cond():
            return
        await pilot.pause(0.05)
    raise TimeoutError


async def record() -> list[tuple[Path, int, str]]:
    frames: list[tuple[Path, int, str]] = []  # (svg, milliseconds on screen, caption)
    app = AIForgeApp(WS)
    async with app.run_test(size=SIZE) as pilot:
        def frame(ms: int, title: str) -> None:
            svg = FRAMES / f"{len(frames):03}.svg"
            app.save_screenshot(str(svg))
            frames.append((svg, ms, title))

        async def type_into(widget: Input, text: str, title: str) -> None:
            for i in range(1, len(text) + 1):
                widget.value = text[:i]
                await pilot.pause(0.02)
                if i % 4 == 0 or i == len(text):
                    frame(120, title)

        app.sub_title = "workspace: ~/AIForge"
        await pilot.pause(0.5)
        frame(1500, "AIForge: build, train and use AI models in the terminal")

        # 1. Create
        await pilot.press("n"); await pilot.pause(0.3)
        await type_into(app.screen.query_one("#name", Input), "spiral-classifier", "1 · Create a model")
        frame(1200, "1 · Create a model")
        app.screen.submit(); await pilot.pause(0.6)

        # 2. Look at the layers
        frame(2200, "2 · Every layer, its shape and parameter count")
        await pilot.press("ctrl+s"); await pilot.pause(0.2)
        await pilot.press("escape"); await pilot.pause(0.4)

        # 3. Train on the built-in spirals
        await pilot.press("t"); await pilot.pause(0.6)
        s = app.screen
        s.query_one("#epochs", Input).value = "40"
        s.query_one("#lr", Input).value = "0.005"
        s.query_one("#load").press()
        await wait_for(pilot, lambda: s.info is not None); await pilot.pause(0.3)
        frame(1500, "3 · Load data: the model needs to fit it")
        await pilot.click("#autofit"); await pilot.pause(0.3)
        frame(900, "3 · Auto-fit model to data")
        await pilot.click("#start")
        shown = 0
        while s.trainer is not None or not s.loss_hist:
            await pilot.pause(0.05)
            if len(s.loss_hist) >= shown + 5:
                shown = len(s.loss_hist)
                frame(350, "3 · Training: loss down, accuracy up")
        await pilot.pause(0.5)
        frame(2500, "3 · Trained: the best epoch is saved")
        await pilot.press("escape"); await pilot.pause(0.4)

        # 4. Use it
        await pilot.press("p"); await pilot.pause(0.6)
        s = app.screen
        await type_into(s.query_one("#raw", Input), "0.1, 0.45", "4 · Use it: get predictions")
        await pilot.press("enter"); await pilot.pause(0.4)
        frame(3500, "4 · Use it: get predictions")
    return frames


def main() -> None:
    if not CHROME:
        sys.exit("Google Chrome or Chromium is needed to render the frames.")
    frames = asyncio.run(record())
    images, durations = [], []
    font = ImageFont.load_default(size=22)
    for svg, ms, caption in frames:
        png = svg.with_suffix(".png")
        svg_to_png(svg, png)
        shot = Image.open(png).convert("RGB")
        shot = shot.resize((WIDTH, round(shot.height * WIDTH / shot.width)), Image.LANCZOS)
        # A caption strip under each frame tells the story without sound.
        im = Image.new("RGB", (WIDTH, shot.height + 44), "#111827")
        im.paste(shot, (0, 0))
        ImageDraw.Draw(im).text((WIDTH // 2, shot.height + 22), caption, font=font, fill="#f9fafb", anchor="mm")
        images.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        durations.append(ms)
    images[0].save(OUT, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"saved {OUT} ({len(images)} frames, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
