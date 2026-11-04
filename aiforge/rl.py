"""Reinforcement learning: built-in environments, and a Deep Q-Network (DQN) learner.

An *agent* (the user's network) looks at the environment's state and outputs one score per action:
how much total reward it expects after taking that action. It acts, gets a reward, and learns from
its own experience; nobody tells it the right move. Plain torch/numpy, no extra dependencies;
independent of the UI.
"""
from __future__ import annotations

import copy
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

REPLAY = 20_000  # experiences remembered
WARMUP = 500  # experiences collected before learning starts
TARGET_EVERY = 250  # steps between copies of the network into the (slow) target network
EVAL_EVERY = 10  # episodes between tests without exploring
EPS_END = 0.05  # exploration left at the end
SMOOTH = 20  # episodes in the moving averages


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


@dataclass
class Step:
    obs: np.ndarray
    reward: float
    done: bool
    timeout: bool = False  # ended by the step limit: not a real end, the future still counts
    event: str = ""  # what happened, in words ("reached the goal", "fell into a trap"...)


class GridWorld:
    """Walk from the start to the goal 🏁 on a square grid without stepping into a trap 🔥.

    The agent sees three layers of size×size cells, flattened: where it is, where the goal is and
    where the traps are. ``layout`` "fixed" keeps the same goal and traps every episode (only the
    start moves); "random" draws a new map each time, which is much harder to learn."""

    name = "gridworld"
    actions = ["up", "down", "left", "right"]
    arrows = ["↑", "↓", "←", "→"]
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    metric = ("Reached the goal", "{:.0%}")  # chart of the share of recent episodes that succeeded
    REWARDS = {"step": -0.1, "wall": -0.3, "trap": -5.0, "goal": 10.0}

    def __init__(self, size: int = 5, traps: int = 3, layout: str = "fixed", max_steps: int = 0,
                 seed: int | None = None):
        if not 3 <= size <= 12:
            raise ValueError("The grid size must be between 3 and 12")
        if not 0 <= traps <= size * size // 3:
            raise ValueError(f"A {size}×{size} grid can have 0 to {size * size // 3} traps")
        self.size, self.n_traps, self.layout = size, traps, layout
        self.max_steps = max_steps or 2 * size * size
        self.rng = np.random.default_rng(seed)
        self.obs_size = 3 * size * size
        self._map = self._new_map(np.random.default_rng(0)) if layout == "fixed" else None
        self.reset()

    def _new_map(self, rng) -> tuple[tuple[int, int], set]:
        """A goal and traps such that every free cell can still reach the goal."""
        n = self.size
        while True:
            cells = [(r, c) for r in range(n) for c in range(n)]
            goal = cells[rng.integers(len(cells))]
            others = [p for p in cells if p != goal]
            traps = {others[i] for i in rng.choice(len(others), self.n_traps, replace=False)}
            if self._all_reach(goal, traps):
                return goal, traps

    def _all_reach(self, goal, traps) -> bool:
        seen, todo = {goal}, [goal]
        while todo:
            r, c = todo.pop()
            for dr, dc in self.moves:
                p = (r + dr, c + dc)
                if 0 <= p[0] < self.size and 0 <= p[1] < self.size and p not in seen and p not in traps:
                    seen.add(p)
                    todo.append(p)
        return len(seen) == self.size * self.size - len(traps)

    def reset(self) -> np.ndarray:
        self.goal, self.traps = self._map if self._map else self._new_map(self.rng)
        free = [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) != self.goal and (r, c) not in self.traps]
        self.pos = free[self.rng.integers(len(free))]
        self.path, self.steps = [self.pos], 0
        return self.observe()

    def observe(self) -> np.ndarray:
        n = self.size
        obs = np.zeros((3, n, n), np.float32)
        obs[0][self.pos] = 1
        obs[1][self.goal] = 1
        for t in self.traps:
            obs[2][t] = 1
        return obs.reshape(-1)

    def step(self, action: int) -> Step:
        dr, dc = self.moves[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1
        reward, done, event = self.REWARDS["step"], False, ""
        if not (0 <= r < self.size and 0 <= c < self.size):
            reward, event = self.REWARDS["wall"], "bumped into the wall"
        else:
            self.pos = (r, c)
            self.path.append(self.pos)
            if self.pos == self.goal:
                reward, done, event = self.REWARDS["goal"], True, "reached the goal"
            elif self.pos in self.traps:
                reward, done, event = self.REWARDS["trap"], True, "fell into a trap"
        timeout = not done and self.steps >= self.max_steps
        if timeout:
            event = "ran out of steps"
        return Step(self.observe(), reward, done or timeout, timeout, event)

    def success(self) -> bool:
        return self.pos == self.goal

    def render(self, path: bool = False) -> str:
        """The grid as text, 2 columns per cell; ``path`` marks the cells visited."""
        walked = set(self.path) if path else set()
        rows = []
        for r in range(self.size):
            line = []
            for c in range(self.size):
                p = (r, c)
                if p == self.pos:  # the agent; at the goal or in a trap it shows what happened
                    cell = "🎉" if p == self.goal else "💥" if p in self.traps else "🤖"
                else:
                    cell = "🏁" if p == self.goal else "🔥" if p in self.traps else "░░" if p in walked else "· "
                line.append(cell)
            rows.append("│" + "".join(line) + "│")
        edge = "─" * (2 * self.size)
        return "\n".join([f"┌{edge}┐", *rows, f"└{edge}┘"])

    def describe(self) -> str:
        r = self.REWARDS
        return (f"GridWorld {self.size}×{self.size}, {self.n_traps} trap{'s' if self.n_traps != 1 else ''}, "
                f"{'the same map every episode' if self.layout == 'fixed' else 'a new map every episode'} · "
                f"rewards: goal +{r['goal']:g}, trap {r['trap']:g}, each step {r['step']:g} · "
                f"at most {self.max_steps} steps")


class CartPole:
    """Keep a pole balanced on a cart by pushing the cart left or right (the classic control task,
    with the same physics as OpenAI Gym's CartPole-v1). +1 reward for every step the pole stays up;
    the episode ends when it tilts past 12° or the cart leaves the track."""

    name = "cartpole"
    actions = ["push left", "push right"]
    arrows = ["←", "→"]
    metric = ("Steps balanced", "{:.0f}")
    GRAVITY, CART, POLE, HALF_LENGTH, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    ANGLE_LIMIT, TRACK = 12 * math.pi / 180, 2.4

    def __init__(self, max_steps: int = 200, seed: int | None = None, **_):
        self.max_steps = max_steps or 200
        self.rng = np.random.default_rng(seed)
        self.obs_size = 4
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(-0.05, 0.05, 4)
        self.steps, self.last_event = 0, ""
        return self.observe()

    def observe(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def step(self, action: int) -> Step:
        x, x_dot, th, th_dot = self.state
        force = self.FORCE if action == 1 else -self.FORCE
        total = self.CART + self.POLE
        pole_ml = self.POLE * self.HALF_LENGTH
        cos, sin = math.cos(th), math.sin(th)
        temp = (force + pole_ml * th_dot ** 2 * sin) / total
        th_acc = (self.GRAVITY * sin - cos * temp) / (
            self.HALF_LENGTH * (4.0 / 3.0 - self.POLE * cos ** 2 / total))
        x_acc = temp - pole_ml * th_acc * cos / total
        x, x_dot = x + self.TAU * x_dot, x_dot + self.TAU * x_acc
        th, th_dot = th + self.TAU * th_dot, th_dot + self.TAU * th_acc
        self.state = np.array([x, x_dot, th, th_dot])
        self.steps += 1
        event = ("the pole fell past 12°" if abs(th) > self.ANGLE_LIMIT else
                 "the cart left the track" if abs(x) > self.TRACK else "")
        done = bool(event)
        timeout = not done and self.steps >= self.max_steps
        if timeout:
            event = f"balanced for all {self.max_steps} steps"
        self.last_event = event
        return Step(self.observe(), 1.0, done or timeout, timeout, event)

    def success(self) -> bool:
        return self.steps >= self.max_steps and abs(self.state[2]) <= self.ANGLE_LIMIT \
            and abs(self.state[0]) <= self.TRACK

    def render(self, path: bool = False) -> str:
        width = 41
        x, _, th, _ = self.state
        col = int(round((max(-self.TRACK, min(self.TRACK, x)) + self.TRACK) / (2 * self.TRACK) * (width - 5))) + 2
        rows = []
        for h in (3, 2, 1):
            off = int(round(math.tan(max(-1.4, min(1.4, th))) * h * 2))
            line = [" "] * width
            p = col + off
            if 0 <= p < width:
                line[p] = "|" if abs(th) < 0.05 else "/" if th > 0 else "\\"
            rows.append("".join(line))
        cart = [" "] * width
        for i, ch in enumerate("[██]"):
            if 0 <= col - 1 + i < width:
                cart[col - 1 + i] = ch
        rows += ["".join(cart), "─" * width,
                 f"angle {math.degrees(th):+5.1f}° · position {x:+.2f} · step {self.steps}/{self.max_steps}"]
        return "\n".join(rows)

    def describe(self) -> str:
        return (f"CartPole · reward +1 for every step the pole stays up · at most {self.max_steps} steps "
                "(balancing all of them is a perfect score)")


ENVS = {"gridworld": GridWorld, "cartpole": CartPole}
ENV_LABELS = {"gridworld": "GridWorld: reach the goal, avoid the traps",
              "cartpole": "CartPole: balance a pole on a moving cart"}


def make_env(cfg: dict, seed: int | None = None):
    name = cfg.get("env", "gridworld")
    if name not in ENVS:
        raise ValueError(f"Unknown environment '{name}' (choose {', '.join(ENVS)})")
    kw = {k: int(cfg[k]) for k in ("size", "traps", "max_steps") if cfg.get(k) not in (None, "")}
    if name == "gridworld" and cfg.get("layout"):
        kw["layout"] = cfg["layout"]
    return ENVS[name](seed=seed, **kw)


def env_cfg(cfg: dict) -> dict:
    """The settings that define an environment, as stored with the model."""
    env = make_env(cfg)
    out = {"type": "env", "env": env.name, "max_steps": env.max_steps}
    if env.name == "gridworld":
        out.update(size=env.size, traps=env.n_traps, layout=env.layout)
    return out


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------


@torch.no_grad()
def q_values(model: nn.Module, obs: np.ndarray) -> torch.Tensor:
    model.eval()
    return model(torch.as_tensor(obs, dtype=torch.float32)[None])[0]


@dataclass
class Episode:
    reward: float = 0.0
    steps: int = 0
    success: bool = False
    event: str = ""
    frame: str = ""  # the last state, drawn (GridWorld: with the path walked)
    actions: list[int] = field(default_factory=list)


def play(model: nn.Module | None, env, explore: float = 0.0, rng=None) -> Episode:
    """One episode: the model's best action each step (or a random one with chance ``explore``;
    no model = always random). Returns what happened."""
    rng = rng or np.random.default_rng()
    obs, ep = env.reset(), Episode()
    while True:
        if model is None or rng.random() < explore:
            a = int(rng.integers(len(env.actions)))
        else:
            a = int(q_values(model, obs).argmax())
        s = env.step(a)
        obs = s.obs
        ep.reward += s.reward
        ep.steps += 1
        ep.actions.append(a)
        if s.done:
            ep.event = s.event
            break
    ep.success = env.success()
    ep.frame = env.render(path=True)
    return ep


def episode_metric(env, ep: Episode) -> float:
    """The number the second chart shows: success (GridWorld) or steps balanced (CartPole)."""
    return float(ep.success) if env.name == "gridworld" else float(ep.steps)


# ---------------------------------------------------------------------------
# Learning: Deep Q-Network
# ---------------------------------------------------------------------------


class Replay:
    """The agent's memory of past steps; learning from random old steps keeps it stable."""

    def __init__(self, size: int, obs_size: int):
        self.obs = torch.zeros(size, obs_size)
        self.next = torch.zeros(size, obs_size)
        self.action = torch.zeros(size, dtype=torch.long)
        self.reward = torch.zeros(size)
        self.end = torch.zeros(size)
        self.size, self.n, self.i = size, 0, 0

    def add(self, obs, action, reward, nxt, end) -> None:
        j = self.i
        self.obs[j], self.next[j] = torch.as_tensor(obs), torch.as_tensor(nxt)
        self.action[j], self.reward[j], self.end[j] = action, reward, float(end)
        self.i, self.n = (j + 1) % self.size, min(self.n + 1, self.size)

    def sample(self, k: int, gen: torch.Generator):
        idx = torch.randint(self.n, (k,), generator=gen)
        return self.obs[idx], self.action[idx], self.reward[idx], self.next[idx], self.end[idx]


def epsilon(episode: int, episodes: int, explore: float) -> float:
    """Chance of a random action: from 100% down to 5% over the first ``explore`` share of the run."""
    span = max(1.0, explore * episodes)
    return max(EPS_END, 1.0 - (1.0 - EPS_END) * (episode - 1) / span)


def train_agent(trainer) -> dict:
    """The DQN loop for ``trainer`` (a trainer.Trainer whose ``preprocess`` holds the environment):
    act (mostly the best known action, sometimes a random one to explore), remember each step, and
    after every step learn from a random batch of remembered steps: the score of an action should
    become *reward now + discount × best score of the next state*. Every few episodes the agent is
    tested without exploring; the best tested version is kept."""
    spec, model, c = trainer.spec, trainer.model, trainer.cfg
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,)))
    env, test_env = make_env(trainer.preprocess, seed), make_env(trainer.preprocess, seed + 1)
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    model.cpu()
    target = copy.deepcopy(model).eval()
    opt = trainer._optimizer()
    memory = Replay(REPLAY, env.obs_size)
    gamma, explore = float(getattr(c, "gamma", 0.99)), float(getattr(c, "explore", 0.5))
    trainer.on_log(f"Device: cpu (small agents learn fastest on the CPU) · {env.describe()} · "
                   f"{sum(p.numel() for p in model.parameters()):,} params")
    started, t_start = time.time(), time.time()
    best, bad, total_steps, recent = -math.inf, 0, 0, deque(maxlen=SMOOTH)
    recent_metric = deque(maxlen=SMOOTH)
    n_test = 10 if env.name == "gridworld" else 3
    for episode in range(1, c.epochs + 1):
        if trainer.stopped:
            break
        t0, eps = time.time(), epsilon(episode, c.epochs, explore)
        obs, ep_reward, losses, steps, s = env.reset(), 0.0, [], 0, None
        while True:
            if trainer.stopped:
                break
            a = int(rng.integers(len(env.actions))) if rng.random() < eps else int(q_values(model, obs).argmax())
            s = env.step(a)
            memory.add(obs, a, s.reward, s.obs, s.done and not s.timeout)
            obs, ep_reward, steps, total_steps = s.obs, ep_reward + s.reward, steps + 1, total_steps + 1
            if memory.n >= max(WARMUP, c.batch_size):
                losses.append(_learn(model, target, opt, memory, c.batch_size, gamma, gen))
            if total_steps % TARGET_EVERY == 0:
                target.load_state_dict(model.state_dict())
            if trainer.on_batch and steps % 10 == 0:
                trainer.on_batch(episode, min(steps, env.max_steps), env.max_steps, ep_reward)
            if s.done:
                break
        if s is None or not s.done:
            break  # a half-played episode isn't recorded
        recent.append(ep_reward)
        recent_metric.append(float(env.success()) if env.name == "gridworld" else float(steps))
        rec = {"epoch": episode, "reward": round(ep_reward, 3), "steps": steps, "epsilon": round(eps, 3),
               "train_loss": float(np.mean(losses)) if losses else 0.0, "avg_reward": float(np.mean(recent)),
               "metric": float(np.mean(recent_metric)), "event": s.event, "log": False}
        if episode % EVAL_EVERY == 0 or episode == c.epochs:
            tests = [play(model, test_env) for _ in range(n_test)]
            rec["val_reward"] = float(np.mean([t.reward for t in tests]))
            rec["val_metric"] = float(np.mean([episode_metric(test_env, t) for t in tests]))
            rec["log"] = True
            if rec["val_reward"] > best:
                best, bad = rec["val_reward"], 0
                trainer.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                rec["best"], rec["frame"] = True, tests[0].frame
            else:
                bad += 1
        rec["seconds"] = round(time.time() - t0, 3)
        trainer.epochs.append(rec)
        if trainer.on_epoch:
            trainer.on_epoch(rec)
        if c.early_stop and bad >= c.early_stop:
            trainer.on_log(f"Early stopping: the tested reward hasn't improved in {bad} tests "
                           f"({bad * EVAL_EVERY} episodes).")
            break
    if trainer.stopped:
        trainer.on_log("Training stopped by user.")
    if trainer.checkpoint:
        trainer.checkpoint.unlink(missing_ok=True)  # nothing to resume: a new run starts quickly
    if trainer.best_state is not None:
        model.load_state_dict(trainer.best_state)
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "duration_s": round(time.time() - t_start, 1),
            "config": asdict(c), "epochs": trainer.epochs,
            "best_score": best if best > -math.inf else None, "stopped": trainer.stopped}


def _learn(model, target, opt, memory: Replay, k: int, gamma: float, gen) -> float:
    obs, action, reward, nxt, end = memory.sample(k, gen)
    model.train()
    q = model(obs).gather(1, action[:, None]).squeeze(1)
    with torch.no_grad():  # double DQN: the network picks the next action, the target scores it
        best_next = model(nxt).argmax(1, keepdim=True)
        goal = reward + gamma * (1 - end) * target(nxt).gather(1, best_next).squeeze(1)
    loss = nn.functional.smooth_l1_loss(q, goal)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    return loss.item()


def run_stats(episodes: list[Episode], env) -> dict:
    rewards = np.array([e.reward for e in episodes])
    steps = np.array([e.steps for e in episodes])
    se = rewards.std(ddof=1) / math.sqrt(len(rewards)) if len(rewards) > 1 else 0.0
    # 95% interval of the average, never beyond what was actually scored
    ci = [max(float(rewards.min()), float(rewards.mean() - 1.96 * se)),
          min(float(rewards.max()), float(rewards.mean() + 1.96 * se))]
    return {"reward": float(rewards.mean()), "reward_ci": ci,
            "reward_min": float(rewards.min()), "reward_max": float(rewards.max()),
            "success": float(np.mean([e.success for e in episodes])), "steps": float(steps.mean()),
            "episodes": len(episodes)}
