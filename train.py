
import json
import argparse

import wandb
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from tempverse.config import Config
from tempverse.shape_data_stream import ShapeDataset
from tempverse.vae import VAE
from tempverse.rev_transformer import RevFormer
from tempverse.trainer import Trainer
from tempverse.utils import create_timestamp


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser("Temporal Modeling on Simple Shapes Rotation")
    parser.add_argument("--configs", nargs='+', default=[])
    args = parser.parse_args()
    
    for i, config_name in enumerate(args.configs, start=1):
        # Load config
        with open(f"configs/{config_name}.json", "r", encoding="utf-8") as json_data:
            config = Config(**json.load(json_data))

        print(f"Processing config {i}/{len(args.configs)}")
        print(f"Project: {config.general.project}, Name: {config.general.name}")

        device = torch.device("cuda")

        data_loader = DataLoader(ShapeDataset(device, config.data), batch_size=1)

        run_timestamp = create_timestamp()
        for j in range(1, config.training.num_reps + 1):
            directory = f"results/{config.general.project}/{run_timestamp}/{config.general.name}/rep_{j}"

            print("{}/{} rep".format(j, config.training.num_reps))

            model = RevFormer(config.rev_transformer, context_size=config.data.context_size, custom_backward=True)
            model.to(device)

            vae_model = VAE(im_channels=config.data.im_channels, config=config.vae).to(device)
            if config.vae.pretrained_vae_path:
                vae_model.eval()
                vae_model.requires_grad_(False)
                vae_model.load_state_dict(torch.load(config.vae.pretrained_vae_path, map_location=device))
            
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
