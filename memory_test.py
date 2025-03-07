import gc
import time
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from calflops import calculate_flops

from tempverse.config import Config, DataConfig, GeneralConfig, TrainingConfig, IntervalModel, TempModelTypes, VaeModelTypes, GradientCalculationWays, TrainTypes
from tempverse.shape_data_stream import MockShapeDataset
from tempverse.vae import VAE
from tempverse.rev_vae import Reversible_MViT_VAE
from tempverse.rev_transformer import RevFormer
from tempverse.vanilla_transformer import VanillaTransformer
from tempverse.pipe_transformer import PipeTransformer
from tempverse.lstm import Seq2SeqLSTM
from tempverse.trainer import Trainer
from tempverse.utils import BaseLogger, seed_everything, create_timestamp, convert_bytes


if __name__ == "__main__":
    logger = BaseLogger(__name__)
    seed_everything()

    parser = argparse.ArgumentParser("Memory Test of Temporal Modeling")
    parser.add_argument("--temp_models", nargs='+', default=["all"])  # some of the ["rev_transformer", "vanilla_transformer", "pipe_transformer", "lstm"]
    parser.add_argument("--vae_models", nargs='+', default=["all"])  # some of the ["rev_vae", "vae"]
    parser.add_argument("--batches", nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--time_steps", nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    args = parser.parse_args()

    train_steps = 1
    config = Config(
        general=GeneralConfig(
            project="", name="", log_to_wandb=False,
            temp_model_type=None, vae_model_type=None
        ),
        data=DataConfig(gradual_complexity=None),
        training=TrainingConfig(
            grad_calc_way=GradientCalculationWays.REVERSE_CALCULATION,
            steps=train_steps,
            record_freq=train_steps + 1,
            print_freq=train_steps + 1,
            save_image_samples_freq=train_steps + 1,
            save_weights_freq=train_steps + 1
        )
    )
    device = torch.device("cuda")

    # prevent usage of shared memory on Windows
    # refer to this: https://github.com/huggingface/pytorch-image-models/discussions/2128
    torch.cuda.set_per_process_memory_fraction(0.95)

    # parse selected temp models
    temp_models = {}
    if "all" in args.temp_models:
        temp_models = {
            "rev_transformer": RevFormer(config.rev_transformer, context_size=config.data.context_size, grad_calc_way=config.training.grad_calc_way).to(device),
            "vanilla_transformer": VanillaTransformer(config.vanilla_transformer, context_size=config.data.context_size).to(device),
            "pipe_transformer": PipeTransformer(config.pipe_transformer, context_size=config.data.context_size).to(device),
            "lstm": Seq2SeqLSTM(config.lstm).to(device),
        }
    else:
        if "rev_transformer" in args.temp_models:
            temp_models["rev_transformer"] = RevFormer(config.rev_transformer, context_size=config.data.context_size, grad_calc_way=config.training.grad_calc_way).to(device)
        if "vanilla_transformer" in args.temp_models:
            temp_models["vanilla_transformer"] = VanillaTransformer(config.vanilla_transformer, context_size=config.data.context_size).to(device)
        if "pipe_transformer" in args.temp_models:
            temp_models["pipe_transformer"] = PipeTransformer(config.pipe_transformer, context_size=config.data.context_size).to(device)
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
    run_timestamp = create_timestamp()
    
    path_dir: str = f"eval_results/memory"
    Path(path_dir).mkdir(parents=True, exist_ok=True)

    memory_results = {}
    # go thought all the combinations
    for temp_model_name, temp_model in temp_models.items():
        for vae_model_name, vae_model in vae_models.items():
            for batch in args.batches:
                for time_step in args.time_steps:
                    for train_type_name, train_type in TrainTypes.__members__.items():
                        seed_everything()
                        config.data.batch_size = batch
                        config.data.time_to_pred = IntervalModel(min=time_step, max=time_step)

                        match train_type:
                            case TrainTypes.DEFAULT:
                                temp_model_key, vae_model_key = temp_model_name, vae_model_name
                                _temp_model, _vae_model = temp_model, vae_model
                                data_loader = DataLoader(MockShapeDataset(device, config.data), batch_size=1)
                            case TrainTypes.VAE_ONLY:
                                temp_model_key, vae_model_key = None, vae_model_name
                                _temp_model, _vae_model = None, vae_model
                                data_loader = DataLoader(MockShapeDataset(device, config.data), batch_size=1)
                            case TrainTypes.TEMP_ONLY:
                                temp_model_key, vae_model_key = temp_model_name, None
                                _temp_model, _vae_model = temp_model, None
                                data_loader = DataLoader(MockShapeDataset(device, config.data, latent=True), batch_size=1)

                        key = f"temp_model={temp_model_key} vae_model={vae_model_key} batch={batch} time_step={time_step}"
                        if key in memory_results:
                            continue

                        logger.info(f"Start working on: {key}")
                        config.training.train_type = train_type
                        trainer = Trainer(
                            _temp_model, _vae_model, device,
                            wandb_runner=None, verbose=False, training_config=config.training,
                        )

                        time_elapsed, memory_allocated = None, None
                        temp_flops, temp_macs, temp_params = None, None, None
                        vae_flops, vae_macs, vae_params = None, None, None
                        try:
                            ts = time.time()
                            trainer.train(start_step=1, data_loader=data_loader)
                            te = time.time()
                            time_elapsed = te - ts

                            memory_allocated = max(0, torch.cuda.max_memory_allocated(device=None) - default_memory_allocated)
                            
                            # Calculate base models information: FLOPS, MACS and parameters count
                            if _temp_model is not None and _vae_model is not None:
                                torch.cuda.empty_cache()
                                input_cal_flops = torch.randn((batch, config.data.context_size, 256, 1, 1), device=device, requires_grad=True)
                                temp_flops, temp_macs, temp_params = calculate_flops(model=_temp_model, kwargs={"x": input_cal_flops, "t": torch.tensor(time_step)}, output_as_string=False, print_results=False)
                                _temp_model.train()
                                del input_cal_flops

                                torch.cuda.empty_cache()
                                input_cal_flops = torch.randn((batch, config.data.im_channels, config.data.render_window_size, config.data.render_window_size), device=device, requires_grad=True)
                                vae_flops, vae_macs, vae_params = calculate_flops(model=_vae_model, kwargs={"x": input_cal_flops}, output_as_string=False, print_results=False)
                                _vae_model.train()
                                del input_cal_flops
                        except RuntimeError as e:
                            if "out of memory" not in str(e) and "not enough memory" not in str(e):
                                raise e

                        # clear cache and update the default memory 
                        trainer.optimizer.zero_grad()
                        del trainer
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device=None)
                        default_memory_allocated = torch.cuda.max_memory_allocated(device=None)

                        memory_info = {
                            "train_consumption": convert_bytes(memory_allocated),
                            "default_consumption": convert_bytes(default_memory_allocated),
                            "temp_flops": temp_flops,
                            "temp_macs": temp_macs,
                            "temp_params": temp_params,
                            "vae_flops": vae_flops,
                            "vae_macs": vae_macs,
                            "vae_params": vae_params,
                            "time_elapsed": time_elapsed,
                        }
                        memory_results[key] = memory_info
                        logger.info(f"Memory info: {memory_info}")
    
                        # save the results
                        with open(f"{path_dir}/{run_timestamp}.json", "w", encoding="utf-8") as f:
                            json.dump(memory_results, f, ensure_ascii=False, indent=4)
