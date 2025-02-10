from enum import Enum

from pydantic import BaseModel, Field


class ExperimentTypes(Enum):
    TEMP_VERSE_FORMER: str = "TEMP_VERSE_FORMER"
    TEMP_VERSE_FORMER_VANILLA_BP: str = "TEMP_VERSE_FORMER_VANILLA_BP"
    VANILLA_TRANSFORMER: str = "VANILLA_TRANSFORMER"
    LSTM: str = "LSTM"


class TemporalPatterns(Enum):
    ACCELERATION: str = "ACCELERATION"
    DECELERATION: str = "DECELERATION"
    OSCILLATION: str = "OSCILLATION"
    INTERRUPTION: str = "INTERRUPTION"


class TrainTypes(Enum):
    DEFAULT: str = "DEFAULT"
    VAE_ONLY: str = "VAE_ONLY"
    TEMP_ONLY: str = "TEMP_ONLY"


class GeneralConfig(BaseModel):
    experiment_type: ExperimentTypes
    project: str
    name: str
    log_to_wandb: bool
    pretrained_temp_model_path: str | None = None
    pretrained_vae_model_path: str | None = None


class IntervalModel(BaseModel):
    min: int
    max: int


class DataConfig(BaseModel):
    temporal_patterns: list[TemporalPatterns] = []
    render_window_size: int = 64
    im_channels: int = 3
    context_size: int = 12
    batch_size: int = 16
    time_to_pred: IntervalModel = IntervalModel(min=1, max=6)

    # Greater than 0
    angle: IntervalModel = IntervalModel(min=5, max=30)
    acceleration_hundredth: IntervalModel = IntervalModel(min=3, max=6)
    oscillation_period: IntervalModel = IntervalModel(min=1, max=6)
    interruption_period: IntervalModel = IntervalModel(min=1, max=6)

    # TODO: Add validators like assert batch_time > context_size

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
    vanilla_transformer: VanillaTransformerConfig = Field(default_factory=VanillaTransformerConfig)
    lstm: LSTM_Config = Field(default_factory=LSTM_Config)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


class ConfigGroup(BaseModel):
    group: list[Config]


if __name__ == "__main__":
    import yaml

    
    ##############################
    ####    GROUP VAE ONLY    ####
    ##############################
    
    vae_only_group = ConfigGroup(
        group=[
            Config(
                general=GeneralConfig(
                    experiment_type=ExperimentTypes.TEMP_VERSE_FORMER,
                    project="temp-verse-former-vae",
                    name="temp-verse-former-vae",
                    log_to_wandb=True
                ),
                data=DataConfig(
                    temporal_patterns=[],
                    time_to_pred=IntervalModel(min=0, max=0),
                    batch_size=64
                ),
                training=TrainingConfig(
                    train_type=TrainTypes.VAE_ONLY
                )
            )
        ]
    )

    with open("configs/vae-only.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            vae_only_group.model_dump(mode="json"),
            f, allow_unicode=True, default_flow_style=None, width=float("inf")
        )

    
    #####################################
    ####    GROUP TempVerseFormer    ####
    #####################################
    
    temp_verse_former_group = ConfigGroup(
        group=[
            Config(
                general=GeneralConfig(
                    experiment_type=ExperimentTypes.TEMP_VERSE_FORMER,
                    project="temp-verse-former",
                    name="temp-verse-former",
                    log_to_wandb=True
                ),
                data=DataConfig(
                    temporal_patterns=[]
                )
            ),
            Config(
                general=GeneralConfig(
                    experiment_type=ExperimentTypes.TEMP_VERSE_FORMER,
                    project="temp-verse-former",
                    name="temp-verse-former--all-patterns",
                    log_to_wandb=True
                ),
                data=DataConfig(
                    temporal_patterns=[TemporalPatterns.ACCELERATION, TemporalPatterns.DECELERATION, TemporalPatterns.OSCILLATION, TemporalPatterns.INTERRUPTION]
                )
            )
        ]
    )

    with open("configs/temp-verse-former.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            temp_verse_former_group.model_dump(mode="json"),
            f, allow_unicode=True, default_flow_style=None, width=float("inf")
        )


    ##############################################
    ####    GROUP TempVerseFormerVanillaBP    ####
    ##############################################
    
    temp_verse_former_vanilla_bp_group = ConfigGroup(
        group=[
            Config(
                general=GeneralConfig(
                    experiment_type=ExperimentTypes.TEMP_VERSE_FORMER_VANILLA_BP,
                    project="temp-verse-former-vanilla-bp",
                    name="temp-verse-former-vanilla-bp",
                    log_to_wandb=True
                ),
                data=DataConfig(
                    temporal_patterns=[]
                )
            ),
            Config(
                general=GeneralConfig(
                    experiment_type=ExperimentTypes.TEMP_VERSE_FORMER_VANILLA_BP,
                    project="temp-verse-former-vanilla-bp",
                    name="temp-verse-former-vanilla-bp--all-patterns",
                    log_to_wandb=True
                ),
                data=DataConfig(
                    temporal_patterns=[TemporalPatterns.ACCELERATION, TemporalPatterns.DECELERATION, TemporalPatterns.OSCILLATION, TemporalPatterns.INTERRUPTION]
                )
            )
        ]
    )

    with open("configs/temp-verse-former-vanilla-bp.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            temp_verse_former_vanilla_bp_group.model_dump(mode="json"),
            f, allow_unicode=True, default_flow_style=None, width=float("inf")
        )


    ########################################
    ####    GROUP VanillaTransformer    ####
    ########################################
    
    temp_verse_vanilla_transformer = ConfigGroup(
        group=[
            Config(
                general=GeneralConfig(
                    experiment_type=ExperimentTypes.VANILLA_TRANSFORMER,
                    project="temp-verse-vanilla-transformer",
                    name="temp-verse-vanilla-transformer",
                    log_to_wandb=True
                ),
                data=DataConfig(
                    temporal_patterns=[]
                )
            ),
            Config(
                general=GeneralConfig(
                    experiment_type=ExperimentTypes.VANILLA_TRANSFORMER,
                    project="temp-verse-vanilla-transformer",
                    name="temp-verse-vanilla-transformer--all-patterns",
                    log_to_wandb=True
                ),
                data=DataConfig(
                    temporal_patterns=[TemporalPatterns.ACCELERATION, TemporalPatterns.DECELERATION, TemporalPatterns.OSCILLATION, TemporalPatterns.INTERRUPTION]
                )
            )
        ]
    )

    with open("configs/temp-verse-vanilla-transformer.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            temp_verse_vanilla_transformer.model_dump(mode="json"),
            f, allow_unicode=True, default_flow_style=None, width=float("inf")
        )


    ##########################
    ####    GROUP LSTM    ####
    ##########################
    
    temp_verse_lstm = ConfigGroup(
        group=[
            Config(
                general=GeneralConfig(
                    experiment_type=ExperimentTypes.LSTM,
                    project="temp-verse-lstm",
                    name="temp-verse-lstm",
                    log_to_wandb=True
                ),
                data=DataConfig(
                    temporal_patterns=[]
                )
            ),
            Config(
                general=GeneralConfig(
                    experiment_type=ExperimentTypes.LSTM,
                    project="temp-verse-lstm",
                    name="temp-verse-lstm--all-patterns",
                    log_to_wandb=True
                ),
                data=DataConfig(
                    temporal_patterns=[TemporalPatterns.ACCELERATION, TemporalPatterns.DECELERATION, TemporalPatterns.OSCILLATION, TemporalPatterns.INTERRUPTION]
                )
            )
        ]
    )

    with open("configs/temp-verse-lstm.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            temp_verse_lstm.model_dump(mode="json"),
            f, allow_unicode=True, default_flow_style=None, width=float("inf")
        )
