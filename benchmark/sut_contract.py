from dataclasses import dataclass


@dataclass
class SUTConfig:
    mode: str
    model_name: str = "qwen3.5:0.8b"
    seed: int = 42
