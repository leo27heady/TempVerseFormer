import json
import yaml
import argparse
from pathlib import Path

import torch
import numpy as np
from einops import rearrange
from torcheval.metrics import Metric, MeanSquaredError
from torcheval.metrics.image import FrechetInceptionDistance
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dotenv import load_dotenv

from tempverse.config import Config, ConfigGroup, TempModelTypes, VaeModelTypes, GradientCalculationWays, TrainTypes
from tempverse.metrics import StructuralSimilarity
from tempverse.shape_data_stream import ShapeDataset
from tempverse.vae import VAE
from tempverse.rev_vae import Reversible_MViT_VAE
from tempverse.rev_transformer import RevFormer
from tempverse.vanilla_transformer import VanillaTransformer
from tempverse.lstm import Seq2SeqLSTM
from tempverse.trainer import Trainer
from tempverse.utils import BaseLogger, seed_everything, create_timestamp


class Validator():

    def __init__(
        self, 
        vae_model, 
        model,
        device,
        save_dir: str
    ) -> None:

        self.vae_model: VAE | Reversible_MViT_VAE = vae_model
        self.model: RevFormer | Seq2SeqLSTM | VanillaTransformer = model
        self.device = device
        self.save_dir = save_dir

        self.metrics: dict[str, Metric] = {
            "mse": MeanSquaredError(device=self.device),
            "fid": FrechetInceptionDistance(device=self.device),
            "ssim": StructuralSimilarity(device=self.device),
        }

    def validate(self, metric: Metric, input: torch.Tensor, target: torch.Tensor) -> float | None:
        if isinstance(metric, MeanSquaredError):
            result = metric.update(
                rearrange(input, "b t c w h -> b (t c w h)"),
                rearrange(target, "b t c w h -> b (t c w h)")
            ).compute().item()
            metric.reset()
        elif isinstance(metric, FrechetInceptionDistance):
            metric.update(
                self.denormalize(rearrange(input, "b t c w h -> (b t) c w h")).clamp(0, 1),
                is_real=False
            )
            metric.update(
                self.denormalize(rearrange(target, "b t c w h -> (b t) c w h")).clamp(0, 1),
                is_real=True
            )
            result = None
        elif isinstance(metric, StructuralSimilarity):
            result = metric.update(
                self.denormalize(rearrange(input, "b t c w h -> (b t) c w h")).clamp(0, 1),
                self.denormalize(rearrange(target, "b t c w h -> (b t) c w h")).clamp(0, 1)
            ).compute().item()
            metric.reset()
        return result

    def denormalize(self, tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
        return tensor * std + mean

    def save_predictions(
        self, x_pred: torch.Tensor, x_expected: torch.Tensor, 
        name: str, time_to_pred: int,
        angles: torch.Tensor, temp_patterns: torch.Tensor
    ) -> None:

        batch, time, channels, height, width = x_pred.shape

        # Directory to save the image series
        save_dir = Path(self.save_dir) / Path("images") / Path(name)
        save_dir.mkdir(parents=True, exist_ok=True)

        x_expected = self.denormalize(x_expected)  # Shape: [batch, time, channels, height, width]
        x_pred = self.denormalize(x_pred)  # Shape: [batch, time, channels, height, width]

        # Save the image series
        concat_image = torch.cat([x_expected, x_pred], dim=4)
        for batch_idx in range(batch):
            series_dir = save_dir / f"batch_{batch_idx}"
            series_dir.mkdir(parents=True, exist_ok=True)

            with open(f"{series_dir}/metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "time_to_pred": time_to_pred,
                        "patterns": temp_patterns[batch_idx].detach().cpu().numpy().tolist(),
                        "angles": angles[batch_idx].detach().cpu().numpy().tolist(),
                    }, 
                    f, ensure_ascii=False, indent=4
                )

            for t in range(time):
                # Save predicted image
                save_path = series_dir / f"t{t:02d}.png"
                save_image(concat_image[batch_idx, t], save_path)

    def run(
        self, 
        images: torch.Tensor, 
        name: str, 
        time_to_pred: int, 
        angles: torch.Tensor, 
        temp_patterns: torch.Tensor
    ) -> dict[str, float | None]:

        batch_size, images_count, c, w, h = images.shape
        context_size = images_count - time_to_pred

        input_images = images[:, :context_size]
        expected_images = images[:, -context_size:]

        z, _ = self.vae_model.encode(rearrange(input_images, "b t c w h -> (b t) c w h"))
        y_pred = self.model(rearrange(z, "(b t) c w h -> b t c w h", b=batch_size, t=context_size), time_to_pred)
        decoder_output = self.vae_model.decode(rearrange(y_pred, "b t c w h -> (b t) c w h"))
        decoder_output = rearrange(decoder_output, "(b t) c w h -> b t c w h", b=batch_size, t=context_size)

        self.save_predictions(
            decoder_output, expected_images,
            name, time_to_pred, angles, temp_patterns
        )

        return {
            name: self.validate(metric, decoder_output, expected_images)
            for name, metric in self.metrics.items()
        }


if __name__ == "__main__":
    logger = BaseLogger(__name__)
    load_dotenv()
    seed_everything()

    parser = argparse.ArgumentParser("Evaluate Temporal Modeling on Simple Shapes Rotation")
    parser.add_argument("--config_groups", nargs='+', default=[])
    args = parser.parse_args()

    # Load config group
    with open(f"configs/{args.config_groups[0]}.yaml", "r", encoding="utf-8") as data:
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
        
        directory = f"eval_results/{run_timestamp}/{config.general.name}"
        validator = Validator(vae_model, model, device, directory)
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
