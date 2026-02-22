from dataclasses import dataclass


@dataclass
class SUTConfig:
    mode: str
    model_name: str = "qwen3:0.6b"
    seed: int = 42
