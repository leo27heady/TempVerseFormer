from enum import Enum

from pydantic import BaseModel, Field

class TemporalPatterns(Enum):
    ACCELERATION: str = "ACCELERATION"
    DECELERATION: str = "DECELERATION"
    OSCILLATION: str = "OSCILLATION"
    INTERRUPTION: str = "INTERRUPTION"


class GeneralConfig(BaseModel):
    project: str = "reversible-transformer"
    name: str = "reversible-transformer--custom-bp"
    log_to_wandb: bool = True


class IntervalModel(BaseModel):
    min: int
    max: int


class DataConfig(BaseModel):
    temporal_patterns: list[TemporalPatterns] = []
    render_window_size: int = 64
    im_channels: int = 3
    context_size: int = 12
    batch_size: int = 64
    time_to_pred: IntervalModel = IntervalModel(min=1, max=6)

    # Greater than 0
    angle: IntervalModel = IntervalModel(min=5, max=30)
    acceleration_hundredth: IntervalModel = IntervalModel(min=3, max=6)
    oscillation_period: IntervalModel = IntervalModel(min=1, max=6)
    interruption_period: IntervalModel = IntervalModel(min=1, max=6)

    # TODO: Add validators like assert batch_time > context_size

class VaeConfig(BaseModel):
    pretrained_vae_path: str | None = None
    z_channels: int = 256
    down_channels: list[int] = [8, 16, 32, 64, 128, 256, 384]
    mid_channels: list[int] = [384]
    down_sample: list[bool] = [True, True, True, True, True, True]
    attn_down: list[bool] = [False, False, False, True, True, True]
    norm_channels: int = 8
    num_heads: int = 4
    num_down_layers: int = 1
    num_mid_layers: int = 2
    num_up_layers: int = 1


class ReverseTransformerConfig(BaseModel):
    input_dim: int = 256
    embed_dim: int = 768
    n_head: int = 8
    depth: int = 8


class TransformerConfig(BaseModel):
    input_dim: int = 256
    embed_dim: int = 768
    n_head: int = 8
    depth: int = 8


class TrainingConfig(BaseModel):
    num_reps: int = 1
    steps: int = 50_000
    lr: float = 5e-6
    weight_decay: float = 0.0
    record_freq: int = 10
    print_freq: int = 10
    save_image_samples_freq: int = 100
    save_weights_freq: int = 1000


class Config(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    vae: VaeConfig = Field(default_factory=VaeConfig)
    rev_transformer: ReverseTransformerConfig = Field(default_factory=ReverseTransformerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


if __name__ == "__main__":
    import json

    config_dict = Config().model_dump(mode="json")
    with open("configs/default.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
