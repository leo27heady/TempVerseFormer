import json
import yaml
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from tempverse.config import Config, ConfigGroup, TempModelTypes, VaeModelTypes, GradientCalculationWays, TrainTypes
from tempverse.shape_data_stream import ShapeDataset
from tempverse.vae_models import VAE, Reversible_MViT_VAE
from tempverse.temp_models import RevFormer, VanillaTransformer, PipeTransformer, Seq2SeqLSTM
from tempverse.validator import Validator
from tempverse.utils import BaseLogger, seed_everything, create_timestamp


if __name__ == "__main__":
    logger = BaseLogger(__name__)
    load_dotenv()
    seed_everything()

    parser = argparse.ArgumentParser("Evaluate Temporal Modeling on Simple Shapes Rotation")
    parser.add_argument("--config_groups", nargs='+', default=[])
    args = parser.parse_args()

    # Load config group
    with open(f"configs/eval/{args.config_groups[0]}.yaml", "r", encoding="utf-8") as data:
        config_group = ConfigGroup(**yaml.safe_load(data))
    
    run_timestamp = create_timestamp()
    for i, config in enumerate(config_group.group, start=1):
        
        seed_everything()
        logger.info(f"Processing config {i}/{len(config_group.group)}")
        logger.info(f"Project: {config.general.project}, Name: {config.general.name}")
        logger.info(f"Temp model: {config.general.temp_model_type}, VAE model: {config.general.vae_model_type}")

        device = torch.device("cuda")

        is_not_efficient_memory = (config.training.grad_calc_way == GradientCalculationWays.VANILLA_BP)
        assert config.general.pretrained_vae_model_path
        assert config.general.pretrained_temp_model_path
        
        model = None
        if config.training.train_type != TrainTypes.VAE_ONLY:
            match config.general.temp_model_type:
                case TempModelTypes.REV_TRANSFORMER:
                    model = RevFormer(config.rev_transformer, context_size=config.data.context_size, grad_calc_way=config.training.grad_calc_way)
                    logger.info(f"RevFormer model with{'out' if is_not_efficient_memory else ''} efficient backward propagation successfully initialized")
                case TempModelTypes.VANILLA_TRANSFORMER:
                    model = VanillaTransformer(config.vanilla_transformer, context_size=config.data.context_size)
                    logger.info("VanillaTransformer model successfully initialized")
                case TempModelTypes.PIPE_TRANSFORMER:
                    model = PipeTransformer(config.pipe_transformer, context_size=config.data.context_size)
                    logger.info("PipeTransformer model successfully initialized")
                case TempModelTypes.LSTM:
                    model = Seq2SeqLSTM(config.lstm)
                    logger.info("Seq2SeqLSTM model successfully initialized")
                case _:
                    error = f"Unknown Temp Model Type: {config.general.temp_model_type}"
                    logger.error(error)
                    raise ValueError(error)
            model.to(device)
            
            # load temp model
            model.load_state_dict(torch.load(config.general.pretrained_temp_model_path, map_location=device))
            model = model.eval().requires_grad_(False)

        match config.general.vae_model_type:
            case VaeModelTypes.VANILLA_VAE:
                vae_model = VAE(im_channels=config.data.im_channels, config=config.vae).to(device)
                logger.info(f"VAE model successfully initialized")
            case VaeModelTypes.REV_VAE:
                vae_model = Reversible_MViT_VAE(im_channels=config.data.im_channels, img_size=config.data.render_window_size, config=config.rev_vae, grad_calc_way=config.training.grad_calc_way).to(device)
                logger.info(f"Rev VAE model with{'out' if is_not_efficient_memory else ''} efficient backward propagation successfully initialized")
            case _:
                error = f"Unknown VAE Model Type: {config.general.vae_model_type}"
                logger.error(error)
                raise ValueError(error)
        
        # load vae model
        vae_model.load_state_dict(torch.load(config.general.pretrained_vae_model_path, map_location=device))
        vae_model = vae_model.eval().requires_grad_(False)
        
        directory = f"eval_results/metrics/{run_timestamp}/{config.general.name}"
        validator = Validator(model, vae_model, device, directory)
        data_loader = DataLoader(ShapeDataset(device, config.data, 0, 1), batch_size=1)

        run_timestamp = create_timestamp()
        results: dict[str, list[float]] = {}
        for step, (batch_t, images, angles, temp_patterns) in enumerate(data_loader, start=1):
            logger.info(f"Start step #{step} processing")
            
            batch_t = batch_t.squeeze(dim=0).to(device=device)
            images = images.squeeze(dim=0).to(device=device)
            angles = angles.squeeze(dim=0).to(device=device)
            temp_patterns = temp_patterns.squeeze(dim=0).to(device=device)
            
            time_to_pred = len(batch_t)

            step_res = validator.run(images, f"{step}", time_to_pred, angles, temp_patterns)
            logger.info(f"Step results {step_res}")
            for metric_name, metric_value in step_res.items():
                if metric_value is None:
                    continue
                results[metric_name] = results.get(metric_name, []) + [metric_value]
            
            if step == config.training.steps:
                break
        
        mean_results = {
            metric_name: np.mean(metric_values)
            for metric_name, metric_values in results.items()
        }
        mean_results["fid"] = validator.metrics["fid"].compute().item()
        validator.metrics["fid"].reset()

        with open(f"{directory}/metrics.json", "w", encoding="utf-8") as f:
            json.dump(mean_results, f, ensure_ascii=False, indent=4)

        del model
        del vae_model
        torch.cuda.empty_cache()
