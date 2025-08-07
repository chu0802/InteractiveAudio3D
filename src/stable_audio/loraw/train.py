from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import random

from stable_audio_tools.models.pretrained import get_pretrained_model

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config

from .network import create_lora_from_config
from .callbacks import LoRAModelCheckpoint, ReLoRAModelCheckpoint
from datetime import datetime

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():
    
    args = get_all_args(defaults_file="stable_audio_config/defaults.ini")

    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    random.seed(seed)
    torch.manual_seed(seed)

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")

    if args.model_config != "":
        with open(args.model_config, "r") as f:
            model_config = json.load(f)
    else:
        with open(args.lora_config, "r") as f:
            lora_config = json.load(f)
        
        model_config.update(lora_config)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    
    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    # LORA: Create and activate
    if args.use_lora == 'true':
        lora = create_lora_from_config(model_config, model)
        if args.lora_ckpt_path:
            lora.load_weights(
                torch.load(args.lora_ckpt_path, map_location="cpu")
            )
        lora.activate()

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    # LORA: Prepare training
    if args.use_lora == 'true':
        lora.prepare_for_training(training_wrapper)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    exp_id = f"{args.object_name}_{args.filter_threshold}_iter{args.iter}_epoch{args.max_epochs}"

    wandb_logger = WandbLogger(project=args.scene_name, id=timestamp)
    wandb_logger.watch(training_wrapper)

    callbacks = []

    callbacks.append(ExceptionCallback())
    
    if args.save_dir:
        checkpoint_dir = os.path.join(args.save_dir, "models", args.scene_name, exp_id) 
    else:
        checkpoint_dir = None

    # LORA: Custom checkpoint callback
    if args.use_lora  == 'true':
        if args.relora_every == 0:
            callbacks.append(LoRAModelCheckpoint(lora=lora, every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1))
        else:
            callbacks.append(ReLoRAModelCheckpoint(lora=lora, every_n_train_steps=args.relora_every, dirpath=checkpoint_dir, save_top_k=-1, checkpoint_every_n_updates=args.checkpoint_every // args.relora_every))
    else:  
        callbacks.append(ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1))

    callbacks.append(ModelConfigEmbedderCallback(model_config))

    callbacks.append(create_demo_callback_from_config(model_config, demo_dl=train_dl))

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    push_wandb_config(wandb_logger, args_dict)

    strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto" 

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
    )
    trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()
