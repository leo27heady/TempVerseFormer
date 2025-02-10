import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from lpips import LPIPS
from einops import rearrange
from torchvision.utils import save_image

from .vae import VAE
from .rev_transformer import RevFormer
from .vanilla_transformer import VanillaTransformer
from .lstm import Seq2SeqLSTM
from .config import TrainingConfig
from .utils import BaseLogger


class Trainer():
    """Class used to train Reverse Transformer.

    Parameters
    ----------
    model :  RevFormer | VanillaTransformer | Seq2SeqLSTM
    vae_model : VAE

    optimizer : torch.optim.Optimizer instance

    device : torch.device

    verbose : bool
        If True prints information (loss, etc) during training.
    """

    def __init__(
        self, model, vae_model, device, wandb_runner, training_config: TrainingConfig,
        verbose=True, save_dir=None
    ):

        self.logger = BaseLogger(__file__)

        self.training_config = training_config
        self.model: RevFormer | VanillaTransformer | Seq2SeqLSTM = model
        self.vae_model: VAE = vae_model
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            [
                {'params': self.model.parameters(), 'lr': self.training_config.lr},
                {'params': self.vae_model.parameters(), 'lr': self.training_config.lr}
            ],
            weight_decay=self.training_config.weight_decay
        )
        
        self.device = device
        self.wandb_runner = wandb_runner
        self.recon_criterion = nn.MSELoss()
        self.save_dir = save_dir
        self.verbose = verbose

        self.lpips_loss_fn = LPIPS(net='alex').to(device=device).requires_grad_(False)

        self.kl_weight = 5e-6
        self.perceptual_weight = 1.0

        self.histories = {
            'steps': [], 'loss_history': [], 'recon_losses_history': [], 'perceptual_losses_history': [], 'kl_losses_history': []
        }
        self.buffer = {
            'loss': [], 'recon_losses': [], 'perceptual_losses': [], 'kl_losses': []
        }


    def denormalize(self, tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
        return tensor * std + mean

    def sample_and_save(
        self, x_pred: torch.Tensor, x_expected: torch.Tensor, 
        num_samples: int, name: str, time_to_pred: int,
        angles: torch.Tensor, temp_patterns: torch.Tensor
    ) -> None:

        time, batch, channels, height, width = x_pred.shape

        # Directory to save the image series
        save_dir = Path(self.save_dir) / Path("sample_images") / Path(name)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Randomly sample N indices from the batch dimension
        indices = torch.randperm(batch)[:num_samples]

        x_pred_sampled = self.denormalize(x_pred[:, indices])  # Shape: [time, num_samples, channels, height, width]
        x_expected_sampled = self.denormalize(x_expected[:, indices])  # Shape: [time, num_samples, channels, height, width]

        # Save the sampled image series
        for i, batch_idx in enumerate(indices):
            series_dir = save_dir / f"batch_{batch_idx.item()}"
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
                concat_image = torch.cat([x_expected_sampled, x_pred_sampled], dim=4)
                # Save predicted image
                save_path = series_dir / f"t{t:02d}.png"
                save_image(concat_image[t, i], save_path)

        self.logger.info(f"Saved {num_samples} image series in '{save_dir}'.")


    def train(self, data_loader):
        """Trains model

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """

        for step, (batch_t, images, angles, temp_patterns) in enumerate(data_loader):
            batch_t = batch_t.squeeze(dim=0).to(device=self.device)
            images = images.squeeze(dim=0).to(device=self.device)
            angles = angles.squeeze(dim=0).to(device=self.device)
            temp_patterns = temp_patterns.squeeze(dim=0).to(device=self.device)

            batch_size, images_count, c, w, h = images.shape

            time_to_pred = len(batch_t)
            context_size = images_count - time_to_pred
            
            self.optimizer.zero_grad()
            
            input_images = images[:, :context_size]
            expected_images = images[:, -context_size:]
            # expected_images = images[:, -time_to_pred:]
            
            input_images = rearrange(input_images, "b t c w h -> (b t) c w h")
            z, encoder_output = self.vae_model.encode(input_images)
            
            z = rearrange(z, "(b t) c w h -> b t c w h", b=batch_size, t=context_size)
            encoder_output = rearrange(encoder_output, "(b t) c w h -> b t c w h", b=batch_size, t=context_size)
            y_pred = self.model(z, time_to_pred)

            decoder_input = rearrange(y_pred, "b t c w h -> (b t) c w h")
            decoder_output = self.vae_model.decode(decoder_input)
            decoder_output = rearrange(decoder_output, "(b t) c w h -> b t c w h", b=batch_size, t=context_size)
            # decoder_output = decoder_output[:, -time_to_pred:]

            recon_loss = self.recon_criterion(decoder_output, expected_images)
            self.buffer['recon_losses'].append(recon_loss.item())

            mean, logvar = torch.chunk(encoder_output, 2, dim=2)
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mean ** 2 - 1 - logvar, dim=[1, 2, 3, 4]))
            self.buffer['kl_losses'].append(kl_loss.item())

            g_loss = recon_loss + (self.kl_weight * kl_loss)

            lpips_loss = torch.mean(self.lpips_loss_fn(
                rearrange(decoder_output, "b t c w h -> (b t) c w h"), 
                rearrange(expected_images, "b t c w h -> (b t) c w h")
            ))
            self.buffer['perceptual_losses'].append(lpips_loss.item())
            
            g_loss += self.perceptual_weight * lpips_loss
            self.buffer['loss'].append(g_loss.item())
            
            g_loss.backward()
            self.optimizer.step()

            pretty_name = f"step{step}-loss{str(g_loss.item()).replace(".", "_")}"
            
            if self.verbose and step % self.training_config.print_freq == 0:
                self.logger.info("\nIteration {}".format(step))
                self.logger.info("Loss: {:.3f}".format(g_loss.item()))

            if step % self.training_config.save_image_samples_freq == 0:
                with torch.no_grad():
                    self.sample_and_save(
                        rearrange(decoder_output, "b t c w h -> t b c w h"), 
                        rearrange(expected_images, "b t c w h -> t b c w h"), 
                        num_samples=2,  # number of batch samples
                        name=pretty_name,
                        time_to_pred=time_to_pred,
                        angles=angles,
                        temp_patterns=temp_patterns,
                    )

            if self.wandb_runner is not None:
                self.wandb_runner.log({
                    "step": step,
                    "time_to_pred": time_to_pred, 
                    "loss": self.buffer['loss'][-1], 
                    "recon_losses": self.buffer['recon_losses'][-1], 
                    "perceptual_losses": self.buffer['perceptual_losses'][-1], 
                    "kl_losses": self.buffer['kl_losses'][-1], 
                })

            # At every record_freq iteration, record mean loss and so on and clear buffer
            if step % self.training_config.record_freq == 0:
                self.histories['steps'].append(step)
                self.histories['loss_history'].append(np.mean(self.buffer['loss']))
                self.histories['recon_losses_history'].append(np.mean(self.buffer['recon_losses']))
                self.histories['perceptual_losses_history'].append(np.mean(self.buffer['perceptual_losses']))
                self.histories['kl_losses_history'].append(np.mean(self.buffer['kl_losses']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['recon_losses'] = []
                self.buffer['perceptual_losses'] = []
                self.buffer['kl_losses'] = []
            
            if step % self.training_config.save_weights_freq == 0:
                save_model_path = Path(self.save_dir) / Path(f"inter_models/{pretty_name}")
                save_model_path.mkdir(parents=True, exist_ok=True)

                torch.save(self.model.state_dict(), save_model_path / "main-model.pt")
                torch.save(self.vae_model.state_dict(), save_model_path / "vae-model.pt")
            
            if step == self.training_config.steps:
                break
