from enum import Enum

from pydantic import BaseModel, Field


class TempModelTypes(Enum):
    REV_TRANSFORMER: str = "REV_TRANSFORMER"
    VANILLA_TRANSFORMER: str = "VANILLA_TRANSFORMER"
    LSTM: str = "LSTM"


class VaeModelTypes(Enum):
    VANILLA_VAE: str = "VANILLA_VAE"
    REV_VAE: str = "REV_VAE"


class GradientCalculationWays(Enum):
    REVERSE_CALCULATION: str = "TEMP_VERSE_FORMER"
    VANILLA_BP: str = "VANILLA_BP"


class TemporalPatterns(Enum):
    ACCELERATION: str = "ACCELERATION"
    DECELERATION: str = "DECELERATION"
    OSCILLATION: str = "OSCILLATION"
    INTERRUPTION: str = "INTERRUPTION"


class TrainTypes(Enum):
    DEFAULT: str = "DEFAULT"
    VAE_ONLY: str = "VAE_ONLY"
    TEMP_ONLY: str = "TEMP_ONLY"


class ResumeTrain(BaseModel):
    wandb_name: str
    wandb_id: str
    resume_folder: str
    temp_model_path: str | None
    vae_model_path: str | None
    step: int

class GeneralConfig(BaseModel):
    project: str
    name: str
    log_to_wandb: bool

    temp_model_type: TempModelTypes | None
    vae_model_type: VaeModelTypes | None
    pretrained_temp_model_path: str | None = None
    pretrained_vae_model_path: str | None = None


class IntervalModel(BaseModel):
    min: int
    max: int


class DataConfig(BaseModel):
    gradual_complexity: list[float] | None = [0.2, 0.3, 0.1, 0.1, 0.3]  # Must be 1 in sum, and length equal to the time_to_pred.max - time_to_pred.min + 1
    temporal_patterns: list[TemporalPatterns] = []
    render_window_size: int = 64
    im_channels: int = 3
    context_size: int = 12
    batch_size: int = 16
    time_to_pred: IntervalModel = IntervalModel(min=1, max=5)

    # Greater than 0
    angle: IntervalModel = IntervalModel(min=5, max=20)
    acceleration_hundredth: IntervalModel = IntervalModel(min=3, max=6)
    oscillation_period: IntervalModel = IntervalModel(min=1, max=6)
    interruption_period: IntervalModel = IntervalModel(min=1, max=6)

    # TODO: Add validators like assert batch_time > context_size

class ReversibleVaeConfig(BaseModel):
    z_channels: int = 256
    encoder_stage_size: int = 4
    encoder_stages: int = 5
    encoder_jump_first_stage: bool = True
    decoder_stage_size: int = 4
    decoder_stages: int = 6
    decoder_halt_final_stage: bool = True
    patch_kernel: list[int] = [4, 4]
    patch_stride: list[int] = [2, 2]
    patch_padding: list[int] = [1, 1]
    patch_embed_dim: int = 8
    num_heads: int = 1
    qkv_pool_kernel: list[int] = [4, 4]
    adaptive_kv_stride: int = 4
    adaptive_window_size: int = 16
    residual_pooling: bool = True
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_path_rate: float = 0.0
    norm_channels: int = 8
    use_abs_pos: bool = False
    use_rel_pos: bool = True
    rel_pos_zero_init: bool = True
    fast_backprop: bool = False

class VaeConfig(BaseModel):
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


class VanillaTransformerConfig(BaseModel):
    input_dim: int = 256
    embed_dim: int = 768
    n_head: int = 8
    depth: int = 8


class LSTM_Config(BaseModel):
    input_dim: int = 256
    embed_dim: int = 768
    n_layers: int = 8
    enc_dropout: float = 0.5
    dec_dropout: float = 0.5


class TrainingConfig(BaseModel):
    train_type: TrainTypes = TrainTypes.DEFAULT
    grad_calc_way: GradientCalculationWays
    num_reps: int = 1
    steps: int = 30_000
    lr: float = 5e-6
    weight_decay: float = 0.0
    record_freq: int = 10
    print_freq: int = 10
    save_image_samples_freq: int = 100
    save_weights_freq: int = 1000


class Config(BaseModel):
    resume_training: ResumeTrain | None = None
    
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    vae: VaeConfig = Field(default_factory=VaeConfig)
    rev_vae: ReversibleVaeConfig = Field(default_factory=ReversibleVaeConfig)
    
    rev_transformer: ReverseTransformerConfig = Field(default_factory=ReverseTransformerConfig)
    vanilla_transformer: VanillaTransformerConfig = Field(default_factory=VanillaTransformerConfig)
    lstm: LSTM_Config = Field(default_factory=LSTM_Config)


class ConfigGroup(BaseModel):
    group: list[Config]


if __name__ == "__main__":
    import yaml

    
    ##############################
    ####    GROUP VAE ONLY    ####
    ##############################
    
    rev_vae_only_group = ConfigGroup(
        group=[
            Config(
                general=GeneralConfig(
                    project="temp-verse-former-rev-vae",
                    name="temp-verse-former-rev-vae",
                    log_to_wandb=True,
                    temp_model_type=None,
                    vae_model_type=VaeModelTypes.REV_VAE,
                ),
                data=DataConfig(
                    temporal_patterns=[],
                    time_to_pred=IntervalModel(min=0, max=0),
                    batch_size=16
                ),
                training=TrainingConfig(
                    train_type=TrainTypes.VAE_ONLY,
                    grad_calc_way=GradientCalculationWays.REVERSE_CALCULATION,
                ),
            )
        ]
    )

    with open("configs/rev-vae-only.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            rev_vae_only_group.model_dump(mode="json"),
            f, allow_unicode=True, default_flow_style=None, width=float("inf")
        )


    #####################################
    ####    GROUP Temp Model ONLY    ####
    #####################################
    
    for temp_model in TempModelTypes.__members__.keys():
        temp_model_name = temp_model.lower().replace("_", "-")
        temp_model_only_group = ConfigGroup(
            group=[
                Config(
                    general=GeneralConfig(
                        project=f"{temp_model_name}-only",
                        name=f"{temp_model_name}-only",
                        log_to_wandb=True,
                        pretrained_vae_model_path="pretrained_vae/vae-model.pt",
                        temp_model_type=temp_model,
                        vae_model_type=VaeModelTypes.REV_VAE,
                    ),
                    data=DataConfig(
                        temporal_patterns=[]
                    ),
                    training=TrainingConfig(
                        steps=25000,
                        train_type=TrainTypes.TEMP_ONLY,
                        grad_calc_way=GradientCalculationWays.REVERSE_CALCULATION
                    )
                ),
                Config(
                    general=GeneralConfig(
                        project=f"{temp_model_name}-only",
                        name=f"{temp_model_name}--all-patterns",
                        log_to_wandb=True,
                        pretrained_vae_model_path="pretrained_vae/vae-model.pt",
                        temp_model_type=temp_model,
                        vae_model_type=VaeModelTypes.REV_VAE,
                    ),
                    data=DataConfig(
                        temporal_patterns=[TemporalPatterns.ACCELERATION, TemporalPatterns.DECELERATION, TemporalPatterns.OSCILLATION, TemporalPatterns.INTERRUPTION]
                    ),
                    training=TrainingConfig(
                        steps=25000,
                        train_type=TrainTypes.TEMP_ONLY,
                        grad_calc_way=GradientCalculationWays.REVERSE_CALCULATION
                    )
                )
            ]
        )

        with open(f"configs/{temp_model_name}-only.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                temp_model_only_group.model_dump(mode="json"),
                f, allow_unicode=True, default_flow_style=None, width=float("inf")
            )


    ##########################
    ####    GROUP FULL    ####
    ##########################
    
    for temp_model in TempModelTypes.__members__.keys():
        temp_model_name = temp_model.lower().replace("_", "-")
        temp_model_group = ConfigGroup(
            group=[
                Config(
                    general=GeneralConfig(
                        project=f"{temp_model_name}",
                        name=f"{temp_model_name}",
                        log_to_wandb=True,
                        temp_model_type=temp_model,
                        vae_model_type=VaeModelTypes.REV_VAE,
                    ),
                    data=DataConfig(
                        temporal_patterns=[]
                    ),
                    training=TrainingConfig(
                        train_type=TrainTypes.DEFAULT,
                        grad_calc_way=GradientCalculationWays.REVERSE_CALCULATION
                    )
                ),
                Config(
                    general=GeneralConfig(
                        project=f"{temp_model_name}",
                        name=f"{temp_model_name}--all-patterns",
                        log_to_wandb=True,
                        temp_model_type=temp_model,
                        vae_model_type=VaeModelTypes.REV_VAE,
                    ),
                    data=DataConfig(
                        temporal_patterns=[TemporalPatterns.ACCELERATION, TemporalPatterns.DECELERATION, TemporalPatterns.OSCILLATION, TemporalPatterns.INTERRUPTION]
                    ),
                    training=TrainingConfig(
                        train_type=TrainTypes.DEFAULT,
                        grad_calc_way=GradientCalculationWays.REVERSE_CALCULATION
                    )
                )
            ]
        )

        with open(f"configs/{temp_model_name}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                temp_model_group.model_dump(mode="json"),
                f, allow_unicode=True, default_flow_style=None, width=float("inf")
            )
