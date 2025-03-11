import json
from pathlib import Path

import torch
from einops import rearrange
from torcheval.metrics import Metric, MeanSquaredError
from torcheval.metrics.image import FrechetInceptionDistance
from torchvision.utils import save_image

from .metrics import StructuralSimilarity
from .config import IntervalModel
from .vae_models import VAE, Reversible_MViT_VAE
from .temp_models import RevFormer, VanillaTransformer, PipeTransformer, Seq2SeqLSTM


class Validator():

    def __init__(
        self, 
        model,
        vae_model, 
        device,
        time_to_pred: IntervalModel,
        save_dir: str
    ) -> None:

        self.model: RevFormer | VanillaTransformer | PipeTransformer | Seq2SeqLSTM = model
        self.vae_model: VAE | Reversible_MViT_VAE = vae_model
        self.device = device
        self.save_dir = save_dir

        self.metrics: dict[str, Metric] = {}
        for t in range(time_to_pred.min, time_to_pred.max + 1):
            self.metrics |= {
                f"mse_{t}": MeanSquaredError(device=self.device),
                f"fid_{t}": FrechetInceptionDistance(device=self.device),
                f"ssim_{t}": StructuralSimilarity(device=self.device),
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

        z, _ = self.vae_model.encode(rearrange(input_images, "b t c w h -> (b t) c w h"))
        y_pred = self.model(rearrange(z, "(b t) c w h -> b t c w h", b=batch_size, t=context_size), time_to_pred)
        decoder_output = self.vae_model.decode(rearrange(y_pred, "b t c w h -> (b t) c w h"))
        decoder_output = rearrange(decoder_output, "(b t) c w h -> b t c w h", b=batch_size, t=y_pred.shape[1])

        if isinstance(self.model, Seq2SeqLSTM) or isinstance(self.model, PipeTransformer):
            decoder_output = decoder_output[:, -time_to_pred:]
            expected_images = images[:, -time_to_pred:]
        else:
            expected_images = images[:, -context_size:]

        self.save_predictions(
            decoder_output, expected_images,
            name, time_to_pred, angles, temp_patterns
        )

        return {
            name: self.validate(metric, decoder_output[:, [-1]], expected_images[:, [-1]])
            for name, metric in self.metrics.items()
            if int(name.split("_")[-1]) == time_to_pred
        }
