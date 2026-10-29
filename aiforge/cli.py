"""Command-line use without the TUI: ``aiforge train``, ``predict``, ``export`` and friends.

Handy on a remote GPU machine, in scripts, or to train overnight. Everything reads and writes
the same workspace as the app, so a model trained here shows up there (and the other way round).
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

out = Console(highlight=False, soft_wrap=True)  # long paths stay copy-pasteable
err = Console(stderr=True, highlight=False, soft_wrap=True)

COMMANDS = ("list", "templates", "new", "train", "predict", "generate", "evaluate", "export", "serve")


class CLIError(Exception):
    """A problem to report in one line, without a traceback."""


def add_commands(parser: argparse.ArgumentParser, common: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", metavar="command",
                                description="Leave it out to open the app. "
                                            "'aiforge <command> -h' shows each command's options.")

    sub.add_parser("list", parents=[common], help="Show the models in the workspace")

    sub.add_parser("templates", parents=[common], help="Show the templates 'aiforge new' can start from")

    p = sub.add_parser("new", parents=[common], help="Create a model from a template")
    p.add_argument("name")
    p.add_argument("-t", "--template", required=True,
                   help="Template name, part of it (e.g. 'tiny gpt') or its number from 'aiforge templates'")

    p = sub.add_parser("train", parents=[common], help="Train (or fine-tune) a model",
                       description="Train a model and save the best epoch. Ctrl+C stops after the current "
                                   "batch and still saves the best epoch so far.")
    p.add_argument("model")
    d = p.add_argument_group("data")
    d.add_argument("-d", "--data",
                   help="A toy set (moons, spirals, circles, blobs, regression), a built-in dataset "
                        "(mnist, fashion_mnist, cifar10, cifar100), a .csv / .txt file or an image folder. "
                        "Leave it out to reuse the data the model was last trained on.")
    d.add_argument("--source", help="Override the guessed data type: csv, csv_text, text_corpus, image_folder")
    d.add_argument("--target", help="CSV column to predict (default: the last one)")
    d.add_argument("--text-column", help="CSV column holding the text (text classifiers)")
    d.add_argument("--tokenizer", choices=["char", "word"], help="How a language model splits text")
    d.add_argument("--val-split", type=float, help="Share of data kept for validation (default 0.2)")
    d.add_argument("--max-samples", type=int, default=0, help="Train on at most this many samples (0 = all)")
    t = p.add_argument_group("training")
    t.add_argument("-e", "--epochs", type=int, default=10)
    t.add_argument("-b", "--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 0.001)")
    t.add_argument("--optimizer", choices=["Adam", "AdamW", "SGD"], default="Adam")
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--scheduler", choices=["none", "cosine", "step"], default="none")
    t.add_argument("--early-stop", type=int, default=0, metavar="EPOCHS",
                   help="Stop after this many epochs without improvement (0 = off)")
    t.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    t.add_argument("--resume", action="store_true", help="Continue from the saved weights")
    t.add_argument("--save-as", metavar="NAME",
                   help="Fine-tune: start from the saved weights and save as a new model")
    t.add_argument("--no-auto-fit", action="store_true",
                   help="Fail instead of adapting the model's input/output to the data")

    p = sub.add_parser("predict", parents=[common], help="Run a trained model on an input or a file")
    p.add_argument("model")
    p.add_argument("input", nargs="?",
                   help="Comma-separated values, a sentence, or an image path (quote it)")
    p.add_argument("-f", "--file", help="Predict every row of a CSV / every image in a folder")
    p.add_argument("-o", "--out", help="Where --file writes its CSV (default: next to the input)")
    p.add_argument("-k", "--top-k", type=int, default=5, help="How many guesses to show")

    p = sub.add_parser("generate", parents=[common], help="Let a language model write text")
    p.add_argument("model")
    p.add_argument("prompt", nargs="?", default="", help="Start of the text (optional)")
    p.add_argument("-n", "--length", type=int, default=200, help="Tokens to write")
    p.add_argument("--temperature", type=float, default=0.8, help="Below 1 is safer, above 1 is wilder")
    p.add_argument("--seed", type=int, help="Same seed, same text")

    p = sub.add_parser("evaluate", parents=[common], help="Measure how good a model really is")
    p.add_argument("model")
    p.add_argument("-f", "--file", help="Evaluate on this file/folder instead of the validation split")
    p.add_argument("--target", help="CSV column with the right answers")
    p.add_argument("--text-column", help="CSV column holding the text")
    p.add_argument("--compare", metavar="MODEL", help="Test whether another model is significantly better")
    p.add_argument("--no-robustness", action="store_true", help="Skip the noise test (faster)")

    p = sub.add_parser("export", parents=[common], help="Export a model for other tools")
    p.add_argument("model")
    p.add_argument("-F", "--format", default="bundle,onnx",
                   help="Comma-separated formats, or 'all' (default: bundle,onnx): bundle, weights_pt, "
                        "weights_safetensors, spec_json, pytorch_py, summary_txt, torchscript, onnx")
    p.add_argument("-o", "--out", help="Folder to write to (default: <workspace>/exports/<model>)")

    p = sub.add_parser("serve", parents=[common], help="Serve a model as a web API with a demo page",
                       description="Start a web server for a trained model: a demo page at /, a JSON API "
                                   "(/predict, /generate, /predict/image) and its docs at /docs. "
                                   "Needs: pip install \"aiforge-studio[serve]\"")
    p.add_argument("model")
    p.add_argument("--host", default="127.0.0.1",
                   help="Address to listen on (default 127.0.0.1: this computer only; 0.0.0.0: your network)")
    p.add_argument("-p", "--port", type=int, default=8000)


def run(args: argparse.Namespace, workspace: Path) -> int:
    from .spec import Workspace

    try:
        ws = Workspace(workspace)
        return globals()[f"cmd_{args.command}"](ws, args) or 0
    except CLIError as e:
        err.print(f"[red]error:[/] {escape(str(e))}")  # messages can hold [brackets], e.g. "[serve]"
        return 1
    except (ValueError, FileNotFoundError, KeyError) as e:
        err.print(f"[red]error:[/] {escape(str(e.args[0] if e.args else e))}")
        return 1


def _need_model(ws, name: str) -> None:
    if not ws.exists(name):
        names = [m["name"] for m in ws.list_models()]
        hint = f" Models here: {', '.join(names)}." if names else " Create one with 'aiforge new'."
        raise CLIError(f"No model called '{name}' in {ws.root}.{hint}")


def _need_weights(ws, name: str) -> None:
    _need_model(ws, name)
    if not ws.has_weights(name):
        raise CLIError(f"'{name}' isn't trained yet - run 'aiforge train {name} --data ...' first")


# -- list / templates / new ---------------------------------------------------------------


def cmd_list(ws, args) -> None:
    rows = ws.list_models()
    if not rows:
        out.print(f"No models in {ws.root} yet. Create one with [b]aiforge new <name> -t <template>[/b].")
        return
    table = Table(box=None, header_style="bold")
    for col in ("Name", "Task", "Params", "Layers", "Trained", "Epochs", "Modified"):
        table.add_column(col, justify="right" if col in ("Params", "Layers", "Epochs") else "left")
    for r in rows:
        table.add_row(r["name"], r["task"], f"{r['params']:,}", str(r["layers"]),
                      "[green]yes[/]" if r["trained"] else "[dim]no[/]", str(r["epochs"]), r["modified"])
    out.print(table)


def _template_names() -> list[tuple[str, str]]:
    from .spec import MODEL_TYPES

    return [(kind, t) for kind, names in MODEL_TYPES.items() for t in names]


def cmd_templates(ws, args) -> None:
    kind_now = None
    for i, (kind, name) in enumerate(_template_names(), 1):
        if kind != kind_now:
            out.print(f"\n[b]{kind}[/b]")
            kind_now = kind
        out.print(f"  {i:>2}. {name}")
    out.print("\nUse one with [b]aiforge new <name> -t <number or part of the name>[/b]")


def find_template(query: str) -> str:
    names = [t for _, t in _template_names()]
    q = query.strip()
    if q.isdigit() and 1 <= int(q) <= len(names):
        return names[int(q) - 1]
    exact = [t for t in names if t.lower() == q.lower()]
    if exact:
        return exact[0]
    words = q.lower().split()
    hits = [t for t in names if all(w in t.lower() for w in words)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise CLIError(f"No template matches '{query}'. See 'aiforge templates'.")
    raise CLIError(f"'{query}' matches {len(hits)} templates: " + "; ".join(hits) + ". Be more specific.")


def cmd_new(ws, args) -> None:
    from .spec import from_template, safe_name

    name = safe_name(args.name)
    if not name:
        raise CLIError("Pick a name with letters or digits")
    if ws.exists(name):
        raise CLIError(f"'{name}' already exists")
    template = find_template(args.template)
    spec = from_template(template, name)
    ws.save_spec(spec)
    out.print(f"[green]✓[/] Created [b]{name}[/b] from '{template}' "
              f"({spec.total_params():,} params). Train it with "
              f"[b]aiforge train {name} --data ...[/b]")


# -- train ---------------------------------------------------------------------------------


def _epoch_line(r: dict) -> str:
    parts = [f"epoch {r['epoch']:>3}", f"loss {r['train_loss']:.4f}"]
    if "train_acc" in r:
        parts.append(f"acc {r['train_acc']:.1%}")
    if "train_ppl" in r:
        parts.append(f"ppl {r['train_ppl']:.2f}")
    if "val_loss" in r:
        parts.append(f"│ val loss {r['val_loss']:.4f}")
    if "val_acc" in r:
        parts.append(f"val acc [b]{r['val_acc']:.1%}[/b]")
    elif "val_mae" in r:
        parts.append(f"val MAE [b]{r['val_mae']:.4g}[/b]")
    elif "val_ppl" in r:
        parts.append(f"val ppl [b]{r['val_ppl']:.2f}[/b]")
    parts.append(f"[dim]{r['seconds']}s[/]")
    return "  ".join(parts)


def cmd_train(ws, args) -> int:
    from . import pipeline
    from .trainer import TrainConfig

    _need_model(ws, args.model)
    spec = ws.load_spec(args.model)
    opts = dict(source=args.source, target=args.target, text_column=args.text_column,
                tokenizer=args.tokenizer, val_split=args.val_split)
    if args.data:
        data_cfg = pipeline.data_cfg_from_arg(args.data, spec, **opts)
    elif spec.data.get("dataset_cfg"):
        data_cfg = {**spec.data["dataset_cfg"], **{k: v for k, v in opts.items() if v is not None}}
    else:
        raise CLIError(f"'{args.model}' hasn't been trained before - say which data with --data")
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, optimizer=args.optimizer,
                      weight_decay=args.weight_decay, scheduler=args.scheduler, early_stop=args.early_stop,
                      device=args.device, max_samples=args.max_samples)
    pipeline.check_config(cfg)

    with out.status("Loading data… (built-in datasets download on first use)"):
        data = pipeline.load_data(ws, spec, data_cfg, args.max_samples)
    info = data.info
    out.print(f"[green]✓[/] {info.description}\n"
              f"  input {tuple(info.input_shape)} {info.input_dtype} · {info.task} · "
              f"{len(data.train):,} train / {len(data.val):,} validation samples")

    progress = Progress(TextColumn("  epoch {task.fields[epoch]}/{task.total}"), BarColumn(),
                        TextColumn("loss {task.fields[loss]:.4f}"), TimeRemainingColumn(),
                        console=out, transient=True)
    bar = progress.add_task("train", total=cfg.epochs, epoch=1, loss=0.0)

    def on_batch(epoch, step, steps, loss):
        progress.update(bar, completed=epoch - 1 + step / steps, epoch=epoch, loss=loss)

    def on_epoch(r):
        progress.console.print(_epoch_line(r))
        if r.get("sample"):
            from rich.markup import escape

            progress.console.print(f"  [dim]✎ sample:[/] [italic]{escape(r['sample'].strip())}[/]")

    trainer_box: list = []
    result: dict = {}

    def work():
        try:
            result["run"], result["spec"] = pipeline.train_model(
                ws, args.model, data, cfg, save_as=args.save_as, resume=args.resume,
                auto_fit=not args.no_auto_fit, on_log=lambda m: progress.console.print(f"[cyan]{m}[/]"),
                on_batch=on_batch, on_epoch=on_epoch, on_trainer=trainer_box.append)
        except Exception as e:  # noqa: BLE001 - reported in the main thread
            result["error"] = e

    # Train in a thread so Ctrl+C can stop it cleanly and still save the best epoch.
    thread = threading.Thread(target=work, daemon=True)
    with progress:
        thread.start()
        while thread.is_alive():
            try:
                thread.join(0.2)
            except KeyboardInterrupt:
                if trainer_box and not trainer_box[0].stopped:
                    trainer_box[0].stop()
                    progress.console.print("[yellow]Stopping after the current batch…[/] "
                                           "(Ctrl+C again to quit without saving)")
                else:
                    raise
    if "error" in result:
        raise result["error"]
    run, spec = result["run"], result["spec"]
    if not run["epochs"]:
        out.print("[yellow]No epoch finished - nothing saved.[/]")
        return 1
    last = min(run["epochs"], key=lambda r: r.get("val_loss", r["train_loss"]))
    best = (f", val acc {last['val_acc']:.1%}" if "val_acc" in last else
            f", val MAE {last['val_mae']:.4g}" if "val_mae" in last else
            f", val perplexity {last['val_ppl']:.2f}" if "val_ppl" in last else "")
    out.print(f"[green]✓[/] Saved [b]{spec.name}[/b] (best epoch {last['epoch']} of "
              f"{len(run['epochs'])}{best}) in {run['duration_s']}s")
    return 0


# -- predict / generate --------------------------------------------------------------------


def cmd_predict(ws, args) -> None:
    from .predict import predict_batch, predict_one

    _need_weights(ws, args.model)
    if not args.file and args.input is None:
        raise CLIError("Give an input (e.g. aiforge predict my-model \"0.1, 0.45\") or --file")
    spec, model, _ = ws.load_model(args.model)
    if args.file:
        path, n = predict_batch(spec, model, args.file, args.out)
        out.print(f"[green]✓[/] Wrote {n:,} predictions to {path}")
        return
    res = predict_one(spec, model, args.input, top_k=args.top_k)
    if spec.task == "regression":
        for label, v in res:
            out.print(f"{label}: [b]{v:.6g}[/b]")
        return
    width = max(len(label) for label, _ in res)
    for i, (label, p) in enumerate(res):
        style = "b" if i == 0 else "dim" if p < 0.05 else ""
        line = f"{label:<{width}}  {p:6.1%}  {'█' * round(p * 30)}"
        out.print(f"[{style}]{line}[/]" if style else line)


def cmd_generate(ws, args) -> None:
    from .textgen import generate

    _need_weights(ws, args.model)
    spec, model, _ = ws.load_model(args.model)
    if spec.task != "language_model":
        raise CLIError(f"'{args.model}' is a {spec.task} model - use 'aiforge predict' instead")
    pre = spec.data.get("preprocess", {})
    if pre.get("type") != "corpus":
        raise CLIError("This language model has no vocabulary saved - train it in AIForge first")
    text = generate(model, pre, args.prompt, length=args.length, temperature=args.temperature, seed=args.seed)
    if args.prompt:
        out.print(args.prompt, end="", style="bold", markup=False)
    out.print(text, markup=False)


# -- evaluate ------------------------------------------------------------------------------


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.4g}"


def cmd_evaluate(ws, args) -> None:
    from . import evaluate as ev
    from .trainer import pick_device

    _need_weights(ws, args.model)
    other = None
    if args.compare:
        _need_weights(ws, args.compare)
    with out.status("Evaluating…") as status:
        spec, model, _ = ws.load_model(args.model)
        ds, ref, desc, orig = ev.load_eval_data(spec, ws.data_dir, "file" if args.file else "val",
                                                args.file or "", args.target, args.text_column)
        if args.compare:
            s2, m2, _ = ws.load_model(args.compare)
            other = (args.compare, s2, m2)
        rep = ev.evaluate_model(spec, model, ds, ref, desc, orig, pick_device("auto"),
                                ws.load_history(args.model), robustness=not args.no_robustness, other=other,
                                progress=lambda stage, d, t: status.update(f"{stage}… {d}/{t}"))
    ev.remember(rep, ws.model_dir(args.model))
    path = ev.save_report(rep, ws.export_dir)

    m, b = rep["metrics"], rep["baseline"]
    out.print(f"[b]{rep['model']}[/b] on {rep['data']}\n")
    out.print(rep["verdict"] + "\n")
    table = Table(box=None, header_style="bold")
    table.add_column("Metric")
    table.add_column("Model", justify="right")
    table.add_column("Baseline", justify="right")
    rows = {"classification": [("Accuracy", "accuracy", True), ("Balanced accuracy", "balanced_accuracy", True),
                               ("Macro F1", "macro_f1", True), ("ROC AUC", "roc_auc", False)],
            "language_model": [("Perplexity", "perplexity", False), ("Token accuracy", "token_accuracy", True)],
            }.get(rep["task"], [("MAE", "mae", False), ("RMSE", "rmse", False), ("R²", "r2", False)])
    for label, key, pct in rows:
        if key in m:
            table.add_row(label, _fmt(m[key], pct), _fmt(b.get(key), pct))
    out.print(table)
    for w in rep.get("warnings", []):
        out.print(f"[yellow]⚠ {w}[/]")
    c = rep.get("comparison")
    if c and c.get("p_value") is not None:
        verdict = "significant" if c["p_value"] < 0.05 else "not significant"
        out.print(f"\nvs {c['other']}: {_fmt(c['score_a'])} vs {_fmt(c['score_b'])} "
                  f"({c['test']}, p = {c['p_value']:.4f}, {verdict})")
    out.print(f"\nFull report: {path}")


# -- serve ---------------------------------------------------------------------------------


def cmd_serve(ws, args) -> None:
    from . import serve

    if problem := serve.missing_dependencies():
        raise CLIError(problem)
    _need_weights(ws, args.model)
    serve.check_servable(ws, args.model)
    if not serve.port_free(args.host, args.port):
        raise CLIError(f"Port {args.port} is already in use - try another one, e.g. -p {args.port + 1}")
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    out.print(f"Serving [b]{args.model}[/b] at [b]{url}[/b]  (API docs: {url}/docs) · Ctrl+C to stop")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        out.print("[yellow]⚠ Anyone on your network can use this model now. There is no password.[/]")
    serve.serve_forever(ws, args.model, args.host, args.port)


# -- export --------------------------------------------------------------------------------


def cmd_export(ws, args) -> None:
    from .model_io import EXPORT_FORMATS, export

    _need_model(ws, args.model)
    fmts = list(EXPORT_FORMATS) if args.format.strip() == "all" else \
        [f.strip() for f in args.format.split(",") if f.strip()]
    unknown = [f for f in fmts if f not in EXPORT_FORMATS]
    if unknown:
        raise CLIError(f"Unknown format {', '.join(unknown)}. Choose from: {', '.join(EXPORT_FORMATS)}")
    if not ws.has_weights(args.model) and any(f not in ("spec_json", "pytorch_py", "summary_txt") for f in fmts):
        out.print(f"[yellow]'{args.model}' isn't trained - exporting random weights.[/]")
    failed = 0
    for f in fmts:
        try:
            path = export(ws, args.model, f, Path(args.out) if args.out else None)
            out.print(f"[green]✓[/] {EXPORT_FORMATS[f][0]:<40} {path}")
        except Exception as e:  # noqa: BLE001 - one failed format shouldn't stop the others
            failed += 1
            out.print(f"[red]✗[/] {EXPORT_FORMATS[f][0]:<40} {e}")
    if failed:
        raise CLIError(f"{failed} of {len(fmts)} formats failed")


def main(argv: list[str] | None = None) -> int:  # for tests: run one command without the TUI
    from .__main__ import build_parser, default_workspace

    args = build_parser().parse_args(argv)
    ws = Path(args.workspace) if args.workspace else default_workspace()
    return run(args, ws.expanduser().resolve())


if __name__ == "__main__":
    sys.exit(main())
