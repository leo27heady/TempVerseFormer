
import json
import yaml
import argparse

import wandb
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from tempverse.config import Config, ConfigGroup, ExperimentTypes, TrainTypes
from tempverse.shape_data_stream import ShapeDataset
from tempverse.vae import VAE
from tempverse.rev_transformer import RevFormer
from tempverse.vanilla_transformer import VanillaTransformer
from tempverse.lstm import Seq2SeqLSTM
from tempverse.trainer import Trainer
from tempverse.utils import BaseLogger, create_timestamp


if __name__ == "__main__":
    logger = BaseLogger(__file__)
    load_dotenv()

    parser = argparse.ArgumentParser("Temporal Modeling on Simple Shapes Rotation")
    parser.add_argument("--config_groups", nargs='+', default=[])
    args = parser.parse_args()

    # Load config group
    with open(f"configs/{args.config_groups[0]}.yaml", "r", encoding="utf-8") as data:
        config_group = ConfigGroup(**yaml.safe_load(data))
    
    for i, config in enumerate(config_group.group, start=1):

        logger.info(f"Processing config {i}/{len(config_group.group)}")
        logger.info(f"Task: {config.general.experiment_type.value}, Project: {config.general.project}, Name: {config.general.name}")

        device = torch.device("cuda")

        data_loader = DataLoader(ShapeDataset(device, config.data), batch_size=1)

        run_timestamp = create_timestamp()
        for j in range(1, config.training.num_reps + 1):
            directory = f"results/{config.general.project}/{run_timestamp}/{config.general.name}/rep_{j}"

            logger.info("{}/{} rep".format(j, config.training.num_reps))

            model = None
            if config.training.train_type != TrainTypes.VAE_ONLY:
                match config.general.experiment_type:
                    case ExperimentTypes.TEMP_VERSE_FORMER:
                        model = RevFormer(config.rev_transformer, context_size=config.data.context_size, custom_backward=True)
                        logger.info("RevFormer model with efficient backward propagation successfully initialized")
                    case ExperimentTypes.TEMP_VERSE_FORMER_VANILLA_BP:
                        model = RevFormer(config.rev_transformer, context_size=config.data.context_size, custom_backward=False)
                        logger.info("RevFormer model with vanilla backward propagation successfully initialized")
                    case ExperimentTypes.VANILLA_TRANSFORMER:
                        model = VanillaTransformer(config.vanilla_transformer, context_size=config.data.context_size)
                        logger.info("VanillaTransformer model successfully initialized")
                    case ExperimentTypes.LSTM:
                        model = Seq2SeqLSTM(config.lstm)
                        logger.info("Seq2SeqLSTM model successfully initialized")
                    case _:
                        error = f"Unknown Experiment Type: {config.general.experiment_type}"
                        logger.error(error)
                        raise ValueError(error)
                model.to(device)

            vae_model = VAE(im_channels=config.data.im_channels, config=config.vae).to(device)
            if config.training.train_type == TrainTypes.TEMP_ONLY:
                vae_model.load_state_dict(torch.load(config.general.pretrained_vae_model_path, map_location=device))
                vae_model = vae_model.eval().requires_grad_(False)

            wandb_runner = None if not config.general.log_to_wandb else wandb.init(
                # set the wandb project where this run will be logged
                project=config.general.project,
                name=f"{config.general.name}--{run_timestamp}--{j}",
                # track hyperparameters and run metadata
                config=config.model_dump(mode="json")
            )
            trainer = Trainer(
                model, vae_model, device,
                wandb_runner=wandb_runner,  
                training_config=config.training,
                verbose=True,
                save_dir=directory
            )
            trainer.train(data_loader)

            with open(f"{directory}/histories.json", "w", encoding="utf-8") as f:
                json.dump(trainer.histories, f, ensure_ascii=False, indent=4)

            if wandb_runner:
                wandb_runner.finish(exit_code=0)
