
import json
import yaml
import argparse

import wandb
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from tempverse.config import Config, ConfigGroup, TempModelTypes, VaeModelTypes, GradientCalculationWays, TrainTypes
from tempverse.shape_data_stream import ShapeDataset
from tempverse.vae import VAE
from tempverse.rev_vae import Reversible_MViT_VAE
from tempverse.rev_transformer import RevFormer
from tempverse.vanilla_transformer import VanillaTransformer
from tempverse.lstm import Seq2SeqLSTM
from tempverse.trainer import Trainer
from tempverse.utils import BaseLogger, create_timestamp


if __name__ == "__main__":
    logger = BaseLogger(__name__)
    load_dotenv()

    parser = argparse.ArgumentParser("Temporal Modeling on Simple Shapes Rotation")
    parser.add_argument("--config_groups", nargs='+', default=[])
    args = parser.parse_args()

    # Load config group
    with open(f"configs/{args.config_groups[0]}.yaml", "r", encoding="utf-8") as data:
        config_group = ConfigGroup(**yaml.safe_load(data))
    
    for i, config in enumerate(config_group.group, start=1):

        logger.info(f"Processing config {i}/{len(config_group.group)}")
        logger.info(f"Project: {config.general.project}, Name: {config.general.name}")
        logger.info(f"Temp model: {config.general.temp_model_type}, VAE model: {config.general.vae_model_type}")

        device = torch.device("cuda")

        run_timestamp = create_timestamp()
        for j in range(1, config.training.num_reps + 1):
            logger.info("{}/{} rep".format(j, config.training.num_reps))

            is_efficient_memory = (config.training.grad_calc_way == GradientCalculationWays.REVERSE_CALCULATION)
            
            model = None
            if config.training.train_type != TrainTypes.VAE_ONLY:
                match config.general.temp_model_type:
                    case TempModelTypes.REV_TRANSFORMER:
                        model = RevFormer(config.rev_transformer, context_size=config.data.context_size, custom_backward=is_efficient_memory)
                        logger.info(f"RevFormer model with{'' if is_efficient_memory else 'out'} efficient backward propagation successfully initialized")
                    case TempModelTypes.VANILLA_TRANSFORMER:
                        model = VanillaTransformer(config.vanilla_transformer, context_size=config.data.context_size)
                        logger.info("VanillaTransformer model successfully initialized")
                    case TempModelTypes.LSTM:
                        model = Seq2SeqLSTM(config.lstm)
                        logger.info("Seq2SeqLSTM model successfully initialized")
                    case _:
                        error = f"Unknown Temp Model Type: {config.general.temp_model_type}"
                        logger.error(error)
                        raise ValueError(error)
                model.to(device)

            match config.general.vae_model_type:
                case VaeModelTypes.VANILLA_VAE:
                    vae_model = VAE(im_channels=config.data.im_channels, config=config.vae).to(device)
                    logger.info(f"VAE model successfully initialized")
                case VaeModelTypes.REV_VAE:
                    vae_model = Reversible_MViT_VAE(im_channels=config.data.im_channels, img_size=config.data.render_window_size, config=config.rev_vae, custom_backward=is_efficient_memory).to(device)
                    logger.info(f"Rev VAE model with{'' if is_efficient_memory else 'out'} efficient backward propagation successfully initialized")
                case _:
                    error = f"Unknown VAE Model Type: {config.general.vae_model_type}"
                    logger.error(error)
                    raise ValueError(error)
            
            if config.training.train_type == TrainTypes.TEMP_ONLY:
                assert config.general.pretrained_vae_model_path
                vae_model.load_state_dict(torch.load(config.general.pretrained_vae_model_path, map_location=device))
                vae_model = vae_model.eval().requires_grad_(False)

                if config.general.pretrained_temp_model_path:
                    model.load_state_dict(torch.load(config.general.pretrained_temp_model_path, map_location=device))
            
            # If resume training, try to restore previous state and continue
            if config.resume_training:
                start_step = config.resume_training.step
                directory = config.resume_training.resume_folder
                wandb_name = config.resume_training.wandb_name
                wandb_id = config.resume_training.wandb_id
                
                if config.resume_training.temp_model_path:
                    model.load_state_dict(torch.load(config.resume_training.temp_model_path, map_location=device))
                
                if config.resume_training.vae_model_path:
                    vae_model.load_state_dict(torch.load(config.resume_training.vae_model_path, map_location=device))
            else:
                start_step = 0
                directory = f"results/{config.general.project}/{run_timestamp}/{config.general.name}/rep_{j}"
                wandb_name = f"{config.general.name}--{run_timestamp}--{j}"
                wandb_id = None

            wandb_runner = None if not config.general.log_to_wandb else wandb.init(
                # set the wandb project where this run will be logged
                project=config.general.project,
                name=wandb_name,
                id=wandb_id,
                resume="must" if config.resume_training else "never",
                # track hyperparameters and run metadata
                config=config.model_dump(mode="json")
            )
            
            data_loader = DataLoader(ShapeDataset(device, config.data, start_step, config.training.steps), batch_size=1)
            
            trainer = Trainer(
                model, vae_model, device,
                wandb_runner=wandb_runner,  
                training_config=config.training,
                verbose=True,
                save_dir=directory
            )
            trainer.train(start_step, data_loader)

            with open(f"{directory}/histories.json", "w", encoding="utf-8") as f:
                json.dump(trainer.histories, f, ensure_ascii=False, indent=4)

            if wandb_runner:
                wandb_runner.finish(exit_code=0)
