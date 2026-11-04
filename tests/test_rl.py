"""Reinforcement learning: the built-in environments, the DQN learner, and an agent's way through
the app (train, use, evaluate, command line, web API)."""
import asyncio

import numpy as np
import pytest
import torch

from aiforge import data as datamod
from aiforge import evaluate as ev
from aiforge import pipeline, rl
from aiforge.cli import main
from aiforge.predict import predict_batch, predict_one
from aiforge.spec import Workspace, from_template
from aiforge.trainer import TrainConfig, Trainer, data_mismatch, fit_spec_to_data


def _agent(ws, template="GridWorld agent (DQN)", name="a", episodes=200, **env):
    torch.manual_seed(0)
    ws.save_spec(from_template(template, name))
    spec = ws.load_spec(name)
    cfg = {"source": "env", **spec.data["env"], **env}
    loaded = pipeline.load_data(ws, spec, cfg)
    run, _ = pipeline.train_model(ws, name, loaded, TrainConfig(epochs=episodes, gamma=0.95),
                                  on_log=lambda m: None)
    spec, model, _ = ws.load_model(name)
    return spec, model, run


# -- environments ---------------------------------------------------------------------------------


def test_gridworld_rules():
    env = rl.GridWorld(size=4, traps=0, seed=0)
    assert env.obs_size == 48 and env.observe().sum() == 2  # the agent and the goal, no traps
    env.pos, env.goal = (0, 0), (0, 2)
    s = env.step(env.actions.index("up"))  # into the wall: stays put
    assert env.pos == (0, 0) and s.reward == rl.GridWorld.REWARDS["wall"] and not s.done
    env.step(env.actions.index("right"))
    s = env.step(env.actions.index("right"))
    assert s.done and not s.timeout and s.reward == rl.GridWorld.REWARDS["goal"] and env.success()
    assert "🎉" in env.render()


def test_gridworld_maps_are_always_solvable():
    env = rl.GridWorld(size=5, traps=8, layout="random", seed=1)
    for _ in range(30):
        env.reset()
        assert env._all_reach(env.goal, env.traps) and env.pos not in env.traps and env.pos != env.goal


def test_gridworld_fixed_map_stays_and_runs_out_of_steps():
    env = rl.GridWorld(size=5, traps=3, seed=2)
    goal, traps = env.goal, env.traps
    env.reset()
    assert (env.goal, env.traps) == (goal, traps)  # only the start moves
    env.pos, env.goal = (0, 0), (4, 4)
    env.traps = set()
    s = None
    for _ in range(env.max_steps):
        s = env.step(env.actions.index("up"))
    assert s.done and s.timeout and s.event == "ran out of steps"
    with pytest.raises(ValueError, match="traps"):
        rl.GridWorld(size=3, traps=5)


def test_cartpole_physics():
    env = rl.CartPole(seed=0)
    assert env.obs_size == 4
    steps = 0
    while not (s := env.step(1)).done:  # always pushing right: the pole falls quickly
        steps += 1
    assert steps < 50 and not s.timeout and not env.success()
    short = rl.CartPole(max_steps=3, seed=0)
    for _ in range(3):
        s = short.step(steps % 2)
    assert s.timeout and short.success()


def test_exploration_fades_out():
    assert rl.epsilon(1, 100, 0.5) == 1.0
    assert rl.epsilon(26, 100, 0.5) == pytest.approx(0.525)
    assert rl.epsilon(80, 100, 0.5) == rl.EPS_END


# -- the model and its data -----------------------------------------------------------------------


def test_environment_is_the_agents_data(tmp_path):
    spec = from_template("CartPole agent (DQN)", "a")
    tr, va, info = datamod.load({"source": "env", "env": "cartpole"}, tmp_path, spec)
    assert len(tr) == len(va) == 0 and info.input_shape == [4] and info.class_names == ["push left", "push right"]
    assert info.output_shape == (2,) and info.preprocess["type"] == "env"
    with pytest.raises(ValueError, match="environment, not a dataset"):
        datamod.load({"source": "synthetic", "kind": "blobs"}, tmp_path, spec)
    assert pipeline.data_cfg_from_arg("gridworld", spec) == {"source": "env", "env": "gridworld"}


def test_auto_fit_sizes_the_agent_for_its_world(tmp_path):
    spec = from_template("GridWorld agent (DQN)", "a")
    _, _, info = datamod.load({"source": "env", "env": "gridworld", "size": 7}, tmp_path, spec)
    assert data_mismatch(spec, info)  # 7×7×3 = 147 inputs, not 75
    fit_spec_to_data(spec, info)
    assert data_mismatch(spec, info) == [] and spec.validate() == [] and spec.input_shape == [147]
    spec.layers[-1].config["out_features"] = 3
    assert "4 actions" in spec.validate()[0]


def test_stop_before_the_first_episode_saves_nothing(tmp_path):
    spec = from_template("GridWorld agent (DQN)", "a")
    tr, va, info = datamod.load({"source": "env", "env": "gridworld"}, tmp_path, spec)
    fit_spec_to_data(spec, info)
    trainer = Trainer(spec, spec.build(), tr, va, TrainConfig(epochs=5), on_log=lambda m: None,
                      preprocess=info.preprocess)
    trainer.stop()
    assert trainer.run()["epochs"] == []


# -- learning and using ---------------------------------------------------------------------------


def test_dqn_learns_gridworld(ws):
    spec, model, run = _agent(ws, episodes=300)  # the Train screen's default; 6 seeds all reached 95%+
    ep = run["epochs"]
    assert len(ep) == 300 and sum(1 for e in ep if "val_reward" in e) == 30  # tested every 10 episodes
    assert np.mean([e["reward"] for e in ep[-20:]]) > np.mean([e["reward"] for e in ep[:20]])
    assert spec.class_names == ["up", "down", "left", "right"] and spec.data["preprocess"]["env"] == "gridworld"
    env = rl.make_env(spec.data["preprocess"], seed=3)
    assert np.mean([rl.play(model, env).success for _ in range(50)]) >= 0.9


def test_predict_scores_every_action(ws):
    spec, model, _ = _agent(ws, episodes=20)
    obs = ",".join(map(str, rl.make_env(spec.data["preprocess"], seed=0).observe()))
    res = predict_one(spec, model, obs, top_k=4)
    assert sorted(a for a, _ in res) == sorted(spec.class_names)
    assert [q for _, q in res] == sorted((q for _, q in res), reverse=True)
    with pytest.raises(ValueError, match="acts in its environment"):
        predict_batch(spec, model, "whatever.csv")


def test_evaluation_report(ws):
    spec, model, _ = _agent(ws)
    rep = ev.evaluate_agent(spec, model, 40, ws.load_history("a"))
    assert rep["metrics"]["success"] > rep["baseline"]["success"]
    lo, hi = rep["metrics"]["reward_ci"]
    assert rep["metrics"]["reward_min"] <= lo <= rep["metrics"]["reward"] <= hi <= rep["metrics"]["reward_max"]
    md = ev.report_markdown(rep)
    for part in ("Random agent", "How the episodes ended", "Which actions it uses", "Best episode", "Learning curve"):
        assert part in md
    ev.remember(rep, ws.model_dir("a"))
    assert ev.summary(ev.json.loads((ws.model_dir("a") / "evaluations.json").read_text())[-1]).startswith("reward")


# -- command line, web API and app ----------------------------------------------------------------


def test_command_line(tmp_path, capsys):
    run = lambda *a: main(["-w", str(tmp_path), *map(str, a)])  # noqa: E731
    assert run("new", "pole", "-t", "cartpole") == 0
    assert run("train", "pole", "-e", 20) == 0  # no --data: the environment its template was made for
    assert run("play", "pole") == 0
    assert run("evaluate", "pole", "--episodes", 5) == 0
    out = capsys.readouterr().out
    assert "tested reward" in out and "Episode 1:" in out and "Random agent" in out
    assert run("train", "pole", "-d", "spirals") == 1  # an agent can't learn from a dataset


def test_web_api(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from aiforge.serve import create_app

    _agent(Workspace(tmp_path), "CartPole agent (DQN)", episodes=10)
    client = TestClient(create_app(Workspace(tmp_path), "a"))
    res = client.post("/predict", json={"input": "0, 0, 0.05, 0"}).json()
    assert {p["label"] for p in res["predictions"]} == {"push left", "push right"}


def test_train_and_use_screens(tmp_path):
    from aiforge.tui.app import AIForgeApp
    from aiforge.tui.predict import PredictScreen
    from aiforge.tui.train import TrainScreen

    ws = Workspace(tmp_path)
    ws.set_setting("star_answered", True)
    ws.save_spec(from_template("GridWorld agent (DQN)", "a"))
    seen = {}

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 50)) as pilot:
            await app.push_screen(TrainScreen("a", finetune=False))
            screen = app.screen
            assert not screen.query_one("#source").display and screen.query_one("#grp-env").display
            screen.query_one("#load").press()
            for _ in range(100):
                await pilot.pause(0.05)
                if screen.info:
                    break
            screen.query_one("#epochs").value = "30"
            screen.action_start()
            for _ in range(600):
                await pilot.pause(0.05)
                if ws.load_history("a") and not screen.trainer:
                    break
            app.pop_screen()
            await app.push_screen(PredictScreen("a"))
            use = app.screen
            await pilot.pause(0.2)
            use.query_one("#step").press()
            await pilot.pause(0.1)
            seen["agent"] = use.steps
            await pilot.press("left")  # your turn
            await pilot.pause(0.1)
            seen["you"] = use.who

    asyncio.run(scenario())
    assert len(ws.load_history("a")[-1]["epochs"]) == 30
    assert seen["agent"] >= 1 and "you" in seen["you"]
