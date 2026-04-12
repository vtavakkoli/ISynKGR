from dataclasses import dataclass


@dataclass
class SUTConfig:
    mode: str
    model_name: str = "gemma4:e2b"
    seed: int = 42
