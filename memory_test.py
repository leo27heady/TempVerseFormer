import gc
import json
import yaml
import argparse
from pathlib import Path

import wandb
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from tempverse.config import Config, ConfigGroup, DataConfig, TrainingConfig, IntervalModel, TempModelTypes, VaeModelTypes, GradientCalculationWays, TrainTypes
from tempverse.shape_data_stream import ShapeDataset
from tempverse.vae import VAE
from tempverse.rev_vae import Reversible_MViT_VAE
from tempverse.rev_transformer import RevFormer
from tempverse.vanilla_transformer import VanillaTransformer
from tempverse.lstm import Seq2SeqLSTM
from tempverse.trainer import Trainer
from tempverse.utils import BaseLogger, seed_everything, create_timestamp


if __name__ == "__main__":
    logger = BaseLogger(__name__)
    seed_everything()

    parser = argparse.ArgumentParser("Memory Test of Temporal Modeling")
    parser.add_argument("--temp_models", nargs='+', default=["all"])  # some of the ["rev_transformer", "vanilla_transformer", "lstm"]
    parser.add_argument("--vae_models", nargs='+', default=["all"])  # some of the ["rev_vae", "vae"]
    parser.add_argument("--batches", nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--time_steps", nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    args = parser.parse_args()

    config = Config(
        data=DataConfig(gradual_complexity=None),
        training=TrainingConfig(
            train_type=TrainTypes.DEFAULT,
            grad_calc_way=GradientCalculationWays.REVERSE_CALCULATION_FULL,
            steps=1,
            record_freq=0,
            print_freq=0,
            save_image_samples_freq=0,
            save_weights_freq=0
        )
    )
    device = torch.device("cuda")

    # parse selected temp models
    temp_models = {}
    if "all" in args.temp_models:
        temp_models = {
            "rev_transformer": RevFormer(config.rev_transformer, context_size=config.data.context_size, grad_calc_way=config.training.grad_calc_way).to(device),
            "vanilla_transformer": VanillaTransformer(config.vanilla_transformer, context_size=config.data.context_size).to(device),
            "lstm": Seq2SeqLSTM(config.lstm).to(device),
        }
    else:
        if "rev_transformer" in args.temp_models:
            temp_models["rev_transformer"] = RevFormer(config.rev_transformer, context_size=config.data.context_size, grad_calc_way=config.training.grad_calc_way).to(device)
        if "vanilla_transformer" in args.temp_models:
            temp_models["vanilla_transformer"] = VanillaTransformer(config.vanilla_transformer, context_size=config.data.context_size).to(device)
        if "lstm" in args.temp_models:
            temp_models["lstm"] = Seq2SeqLSTM(config.lstm).to(device)
        assert temp_models
    
    # parse selected vae models
    vae_models = {}
    if "all" in args.vae_models:
        vae_models = {
            "rev_vae": Reversible_MViT_VAE(im_channels=config.data.im_channels, img_size=config.data.render_window_size, config=config.rev_vae, grad_calc_way=config.training.grad_calc_way).to(device),
            "vae": VAE(im_channels=config.data.im_channels, config=config.vae).to(device),
        }
    else:
        if "rev_vae" in args.temp_models:
            temp_models["rev_vae"] = Reversible_MViT_VAE(im_channels=config.data.im_channels, img_size=config.data.render_window_size, config=config.rev_vae, grad_calc_way=config.training.grad_calc_way).to(device)
        if "vae" in args.temp_models:
            temp_models["vae"] = VAE(im_channels=config.data.im_channels, config=config.vae).to(device)
        assert vae_models
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=None)
    default_memory_allocated = torch.cuda.max_memory_allocated(device=None)
    logger.info(f"Default memory allocated: {default_memory_allocated}")
    run_timestamp = create_timestamp()

    memory_results = []
    # go thought all the combinations
    for temp_model_name, temp_model in temp_models.items():
        for vae_model_name, vae_model in vae_models.items():
            for batch in args.batches:
                for time_step in args.time_steps:
                    seed_everything()
                    config.data.batch_size = batch
                    config.data.time_to_pred = IntervalModel(min=time_step, max=time_step)
                    data_loader = DataLoader(ShapeDataset(device, config.data, 0, 1), batch_size=1)
                    trainer = Trainer(
                        temp_model, vae_model, device,
                        wandb_runner=None,  
                        training_config=config.training,
                        verbose=False,
                    )

                    try:
                        trainer.train(start_step=1, data_loader=data_loader)
                        memory_allocated = torch.cuda.max_memory_allocated(device=None)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            memory_allocated = None
                        else:
                            raise e

                    memory_results.append({
                        "temp_model": temp_model_name,
                        "vae_model": vae_model_name,
                        "batch":  batch,
                        "time_step": time_step,
                        "memory_consumption": memory_allocated
                    })

                    trainer.optimizer.zero_grad()
                    del trainer
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device=None)
    
    # save the results
    path_dir: str = f"eval_results/memory"
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{path_dir}/{run_timestamp}.jsonl", "w", encoding="utf-8") as f:
        for entry in memory_results:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
