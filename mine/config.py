"""Strict, separately editable TOML configuration (paths are relative to the working directory)."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class EnvConfig:
    n_games: int = 8
    seat: int = 0
    threads: int = 4
    rules: dict = field(default_factory=lambda: {"turn_action_caps": [48]})


@dataclass
class ModelConfig:
    embedding_dim: int = 16
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])


@dataclass
class PPOConfig:
    total_timesteps: int = 131072
    rollout_steps: int = 256
    learning_rate: float = 0.0003
    anneal_lr: bool = True
    gamma: float = 0.999
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    value_clip: bool = True
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03


@dataclass
class EvalConfig:
    episodes: int = 16
    n_games: int = 4
    seed: int = 1000000
    every_updates: int = 8
    deterministic: bool = False
    max_vector_steps: int = 20000


@dataclass
class RunConfig:
    seed: int = 42
    device: str = "cpu"
    torch_threads: int = 2
    output_dir: str = "mine/runs/ppo"
    checkpoint_every_updates: int = 8


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        classes = {"env": EnvConfig, "model": ModelConfig, "ppo": PPOConfig,
                   "evaluation": EvalConfig, "run": RunConfig}
        unknown = set(raw) - set(classes)
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        sections = {}
        for name, kind in classes.items():
            values = raw.get(name, {})
            unknown = set(values) - {f.name for f in fields(kind)}
            if unknown:
                raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
            sections[name] = kind(**values)
        config = cls(**sections)
        config.validate()
        return config

    def validate(self):
        positive = {
            "env.n_games": self.env.n_games, "env.threads": self.env.threads,
            "model.embedding_dim": self.model.embedding_dim,
            "ppo.total_timesteps": self.ppo.total_timesteps,
            "ppo.rollout_steps": self.ppo.rollout_steps,
            "ppo.update_epochs": self.ppo.update_epochs,
            "ppo.minibatch_size": self.ppo.minibatch_size,
            "evaluation.episodes": self.evaluation.episodes,
            "evaluation.n_games": self.evaluation.n_games,
            "evaluation.every_updates": self.evaluation.every_updates,
            "evaluation.max_vector_steps": self.evaluation.max_vector_steps,
            "run.torch_threads": self.run.torch_threads,
            "run.checkpoint_every_updates": self.run.checkpoint_every_updates,
        }
        for name, value in positive.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for section, names in ((self.ppo, ("anneal_lr", "value_clip")),
                               (self.evaluation, ("deterministic",))):
            for name in names:
                if type(getattr(section, name)) is not bool:
                    raise ValueError(f"{name} must be a boolean")
        if not self.model.hidden_sizes or any(type(n) is not int or n <= 0
                                             for n in self.model.hidden_sizes):
            raise ValueError("model.hidden_sizes must contain positive integers")
        if type(self.env.seat) is not int or not 0 <= self.env.seat < 8:
            raise ValueError("env.seat must be 0..7")
        for value in (self.run.seed, self.evaluation.seed):
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError("Seeds must be integers in 0..2**32-1")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_coef", "value_coef",
                     "entropy_coef", "max_grad_norm", "target_kl"):
            value = getattr(self.ppo, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"ppo.{name} must be finite")
        if not 0 < self.ppo.gamma <= 1 or not 0 <= self.ppo.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1], gae_lambda in [0, 1]")
        if not 0 < self.ppo.clip_coef < 1:
            raise ValueError("clip_coef must be in (0, 1)")
        if min(self.ppo.learning_rate, self.ppo.max_grad_norm) <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if min(self.ppo.value_coef, self.ppo.entropy_coef, self.ppo.target_kl) < 0:
            raise ValueError("Loss coefficients and target_kl must be non-negative")
        batch = self.env.n_games * self.ppo.rollout_steps
        if batch < 2 or self.ppo.minibatch_size < 2 or batch % self.ppo.minibatch_size:
            raise ValueError("rollout batch must be divisible by minibatch_size (both >= 2)")
        if self.ppo.total_timesteps % batch:
            raise ValueError("total_timesteps must be divisible by n_games * rollout_steps")
        if self.env.rules.get("num_players", 8) != 8:
            raise ValueError("This baseline requires eight-player games")
        if self.env.rules.get("experimental_mode", False):
            raise ValueError("experimental_mode has actions outside the fixed action space")
        caps = self.env.rules.get("turn_action_caps", [])
        if not isinstance(caps, list) or not caps or any(type(x) is not int or x < 1 for x in caps):
            raise ValueError("env.rules.turn_action_caps must contain positive action limits")
        if not isinstance(self.run.output_dir, str) or not self.run.output_dir:
            raise ValueError("run.output_dir must be a nonempty path")


def load_config(path):
    with Path(path).open("rb") as file:
        return Config.from_dict(tomllib.load(file))
