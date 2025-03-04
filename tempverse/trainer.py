import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from lpips import LPIPS
from einops import rearrange
from torchvision.utils import save_image

from .vae import VAE
from .rev_vae import Reversible_MViT_VAE
from .rev_transformer import RevFormer
from .vanilla_transformer import VanillaTransformer
from .lstm import Seq2SeqLSTM
from .config import TrainingConfig, TrainTypes
from .utils import BaseLogger


class Trainer():
    """Class used to train Reverse Transformer.

    Parameters
    ----------
    model :  RevFormer | VanillaTransformer | Seq2SeqLSTM
    vae_model : VAE | Reversible_MViT_VAE

    optimizer : torch.optim.Optimizer instance

    device : torch.device

    verbose : bool
        If True prints information (loss, etc) during training.
    """

    def __init__(
        self, model, vae_model, device, wandb_runner, training_config: TrainingConfig,
        verbose=True, save_dir=None
    ):

        self.logger = BaseLogger(__name__)
        self.training_config = training_config

        self.model: RevFormer | VanillaTransformer | Seq2SeqLSTM | None = model
        self.vae_model: VAE | Reversible_MViT_VAE = vae_model

        match self.training_config.train_type:
            case TrainTypes.DEFAULT:
                params = [
                    {'params': self.model.parameters(), 'lr': self.training_config.lr},
                    {'params': self.vae_model.parameters(), 'lr': self.training_config.lr}
                ]
            case TrainTypes.VAE_ONLY:
                params = [
                    {'params': self.vae_model.parameters(), 'lr': self.training_config.lr}
                ]
            case TrainTypes.TEMP_ONLY:
                params = [
                   {'params': self.model.parameters(), 'lr': self.training_config.lr},
                ]
            case _:
                raise ValueError(f"Unknown Training Type: {self.training_config.train_type}")

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(params, weight_decay=self.training_config.weight_decay)
        self.device = device
        self.wandb_runner = wandb_runner
        self.recon_criterion = nn.MSELoss()
        self.save_dir = save_dir
        self.verbose = verbose

        if self.training_config.train_type != TrainTypes.TEMP_ONLY:
            self.lpips_loss_fn = LPIPS(net='alex').to(device=device).requires_grad_(False)

        self.kl_weight = 1.0
        self.max_kl_weight = 1e-4
        self.perceptual_weight = 1.0

        self.histories = {
            'steps': [], 'loss_history': [], 'recon_losses_history': [], 'vae_recon_losses_history': [], 'perceptual_losses_history': [], 'kl_losses_history': []
        }
        self.buffer = {
            'loss': [], 'recon_losses': [], 'vae_recon_losses': [], 'perceptual_losses': [], 'kl_losses': []
        }

    def warmup(self, current_step: int):
        warmup_step = self.training_config.kl_weight_warmup_steps + self.training_config.kl_weight_start_step
        if current_step < self.training_config.kl_weight_start_step:
            self.kl_weight = 0.0
        elif current_step < warmup_step:
            self.kl_weight = self.max_kl_weight * float(current_step / warmup_step)
        else:
            self.kl_weight = self.max_kl_weight

    def denormalize(self, tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
        return tensor * std + mean

    def sample_and_save(
        self, x_pred: torch.Tensor, x_expected: torch.Tensor, 
        num_samples: int, name: str, time_to_pred: int,
        angles: torch.Tensor, temp_patterns: torch.Tensor
    ) -> None:

        batch, time, channels, height, width = x_pred.shape

        # Directory to save the image series
        save_dir = Path(self.save_dir) / Path("sample_images") / Path(name)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Randomly sample N indices from the batch dimension
        indices = torch.randperm(batch)[:num_samples]

        x_pred_sampled = self.denormalize(x_pred[indices])  # Shape: [batch, time, channels, height, width]
        x_expected_sampled = self.denormalize(x_expected[indices])  # Shape: [batch, time, channels, height, width]

        # Save the sampled image series
        concat_image = torch.cat([x_expected_sampled, x_pred_sampled], dim=4)
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
                # Save predicted image
                save_path = series_dir / f"t{t:02d}.png"
                save_image(concat_image[i, t], save_path)

        self.logger.info(f"Saved {num_samples} image series in '{save_dir}'.")


    def train(self, start_step, data_loader):
        """Trains model

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """

        for step, (batch_t, images, angles, temp_patterns) in enumerate(data_loader, start=start_step):
            self.warmup(step)

            batch_t = batch_t.squeeze(dim=0).to(device=self.device)
            images = images.squeeze(dim=0).to(device=self.device)
            angles = angles.squeeze(dim=0).to(device=self.device)
            temp_patterns = temp_patterns.squeeze(dim=0).to(device=self.device)
            
            time_to_pred = len(batch_t)
            batch_size, images_count, c, w, h = images.shape
            context_size = images_count - time_to_pred
            input_images = images[:, :context_size]

            if isinstance(self.model, Seq2SeqLSTM):
                expected_images = images[:, -time_to_pred:]
            else:
                expected_images = images[:, -context_size:]
            # expected_images = images[:, -time_to_pred:]

            self.optimizer.zero_grad()
            
            # Run selected training type
            match self.training_config.train_type:
                case TrainTypes.DEFAULT:
                    loss, decoder_output = self.train_default(input_images, expected_images, time_to_pred)
                case TrainTypes.VAE_ONLY:
                    loss, decoder_output = self.train_vae_only(images)
                case TrainTypes.TEMP_ONLY:
                    loss, temp_output = self.train_temp_only(images, context_size, time_to_pred)
            
            self.buffer['loss'].append(loss.item())
            
            loss.backward()
            self.optimizer.step()

            pretty_name = f"step{step}-loss{str(loss.item()).replace(".", "_")}"
            
            if self.verbose and step % self.training_config.print_freq == 0:
                self.logger.info("Iteration {}".format(step))
                self.logger.info("Loss: {:.3f}".format(loss.item()))

            if step % self.training_config.save_image_samples_freq == 0:
                if self.training_config.train_type == TrainTypes.TEMP_ONLY:
                    decoder_input = rearrange(temp_output, "b t c w h -> (b t) c w h")
                    decoder_output = self.vae_model.decode(decoder_input)
                    decoder_output = rearrange(decoder_output, "(b t) c w h -> b t c w h", b=batch_size, t=context_size)

                with torch.no_grad():
                    self.sample_and_save(
                        decoder_output, 
                        expected_images, 
                        num_samples=2,  # number of batch samples
                        name=pretty_name,
                        time_to_pred=time_to_pred,
                        angles=angles,
                        temp_patterns=temp_patterns,
                    )

            if self.wandb_runner is not None:
                params = {
                    "step": step,
                    "time_to_pred": time_to_pred,
                    "loss": self.buffer['loss'][-1]
                }
                
                if self.buffer['recon_losses']:
                    params["recon_losses"] = self.buffer['recon_losses'][-1]
                
                if self.buffer['vae_recon_losses']:
                    params["vae_recon_losses"] = self.buffer['vae_recon_losses'][-1]
                
                if self.buffer['perceptual_losses']:
                    params["perceptual_losses"] = self.buffer['perceptual_losses'][-1]
                
                if self.buffer['kl_losses']:
                    params["kl_losses"] = self.buffer['kl_losses'][-1]
                
                self.wandb_runner.log(params)

            # At every record_freq iteration, record mean loss and so on and clear buffer
            if step % self.training_config.record_freq == 0:
                self.histories['steps'].append(step)
                self.histories['loss_history'].append(np.mean(self.buffer['loss']))
                self.histories['recon_losses_history'].append(np.mean(self.buffer['recon_losses']))
                self.histories['vae_recon_losses_history'].append(np.mean(self.buffer['vae_recon_losses']))
                self.histories['perceptual_losses_history'].append(np.mean(self.buffer['perceptual_losses']))
                self.histories['kl_losses_history'].append(np.mean(self.buffer['kl_losses']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['recon_losses'] = []
                self.buffer['vae_recon_losses'] = []
                self.buffer['perceptual_losses'] = []
                self.buffer['kl_losses'] = []
            
            if step % self.training_config.save_weights_freq == 0:
                save_model_path = Path(self.save_dir) / Path(f"inter_models/{pretty_name}")
                save_model_path.mkdir(parents=True, exist_ok=True)
                
                match self.training_config.train_type:
                    case TrainTypes.DEFAULT:
                        torch.save(self.model.state_dict(), save_model_path / "main-model.pt")
                        torch.save(self.vae_model.state_dict(), save_model_path / "vae-model.pt")
                    case TrainTypes.VAE_ONLY:
                        torch.save(self.vae_model.state_dict(), save_model_path / "vae-model.pt")
                    case TrainTypes.TEMP_ONLY:
                        torch.save(self.model.state_dict(), save_model_path / "main-model.pt")

            if step == self.training_config.steps:
                break
    
    def calculate_kl_loss(self, encoder_output):
        mean, logvar = torch.chunk(encoder_output, 2, dim=1)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3]))
        self.buffer['kl_losses'].append(kl_loss.item())
        return kl_loss
    
    def train_default(self, input_images, expected_images, time_to_pred):

        batch_size, context_size, c, w, h = input_images.shape

        z, encoder_output = self.vae_model.encode(rearrange(input_images, "b t c w h -> (b t) c w h"))
        t0_decoder_output = self.vae_model.decode(z)
        
        t0_decoder_output = rearrange(t0_decoder_output, "(b t) c w h -> b t c w h", b=batch_size, t=context_size)
        
        z = rearrange(z, "(b t) c w h -> b t c w h", b=batch_size, t=context_size)
        y_pred = self.model(z, time_to_pred)

        decoder_input = rearrange(y_pred, "b t c w h -> (b t) c w h")
        decoder_output = self.vae_model.decode(decoder_input)
        decoder_output = rearrange(decoder_output, "(b t) c w h -> b t c w h", b=batch_size, t=y_pred.shape[1])
        # decoder_output = decoder_output[:, -time_to_pred:]

        recon_loss = self.recon_criterion(decoder_output, expected_images)
        self.buffer['recon_losses'].append(recon_loss.item())

        vae_recon_loss = self.recon_criterion(t0_decoder_output, input_images)
        self.buffer['vae_recon_losses'].append(vae_recon_loss.item())

        loss = recon_loss + vae_recon_loss

        kl_loss = self.calculate_kl_loss(encoder_output)
        loss += self.kl_weight * kl_loss

        lpips_loss = torch.mean(self.lpips_loss_fn(
            rearrange(decoder_output, "b t c w h -> (b t) c w h"), 
            rearrange(expected_images, "b t c w h -> (b t) c w h")
        ))
        self.buffer['perceptual_losses'].append(lpips_loss.item())
        
        loss += self.perceptual_weight * lpips_loss

        return loss, decoder_output
    
    def train_vae_only(self, images):

        batch_size, images_count, c, w, h = images.shape

        images = rearrange(images, "b t c w h -> (b t) c w h")
        decoder_output, encoder_output = self.vae_model(images)

        recon_loss = self.recon_criterion(decoder_output, images)
        self.buffer['vae_recon_losses'].append(recon_loss.item())

        kl_loss = self.calculate_kl_loss(encoder_output)
        loss = recon_loss + (self.kl_weight * kl_loss)

        lpips_loss = torch.mean(self.lpips_loss_fn(decoder_output, images))
        self.buffer['perceptual_losses'].append(lpips_loss.item())
        
        loss += self.perceptual_weight * lpips_loss

        decoder_output = rearrange(decoder_output, "(b t) c w h -> b t c w h", b=batch_size, t=images_count)
        return loss, decoder_output

    def train_temp_only(self, images, context_size, time_to_pred):

        batch_size, images_count, c, w, h = images.shape

        images = rearrange(images, "b t c w h -> (b t) c w h")
        with torch.no_grad():
            z, _ = self.vae_model.encode(images)
        
        z = rearrange(z, "(b t) c w h -> b t c w h", b=batch_size, t=images_count)
        y_pred = self.model(z[:, :context_size], time_to_pred)

        loss = self.recon_criterion(y_pred, z[:, -context_size:])
        self.buffer['recon_losses'].append(loss.item())

        return loss, y_pred
