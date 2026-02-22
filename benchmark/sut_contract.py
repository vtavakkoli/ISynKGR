from dataclasses import dataclass


@dataclass
class SUTConfig:
    mode: str
    model_name: str = "Qwen/Qwen3-0.6B"
    seed: int = 42
