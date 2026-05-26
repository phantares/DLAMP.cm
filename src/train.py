from pathlib import Path
from dotenv import dotenv_values
import hydra
from lightning import Trainer
import torch
from omegaconf import DictConfig

from utils import write_wandb_id, load_wandb_id


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg) -> None:
    env = dotenv_values(".env")

    dtype = getattr(torch, cfg.dtype, torch.float32)
    torch.set_default_dtype(dtype)

    experiment_name = cfg.experiment.name
    print(f"Training experiment: {experiment_name}")

    datamodule = hydra.utils.instantiate(
        cfg.dataset, input_dir=Path(env.get("INPUT_DIR")), dtype=dtype
    )

    model = hydra.utils.instantiate(
        cfg.model.system,
        single_channel=len(cfg.dataset.var.input_single)
        + len(cfg.dataset.var.input_static),
        upper_channel=len(cfg.dataset.var.input_upper),
        output_channel=len(cfg.dataset.var.target),
    )
    model = model.to(dtype)

    ckpt_dir = Path(cfg.callbacks.model_checkpoint.dirpath)
    wandb_id_file = ckpt_dir / env.get("WANDB_ID_FILE")
    exist_run_id = load_wandb_id(wandb_id_file)
    logger = hydra.utils.instantiate(
        cfg.logger, **({"id": exist_run_id, "resume": "must"} if exist_run_id else {})
    )

    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    gpu_count = torch.cuda.device_count()
    strategy = "ddp" if gpu_count > 1 else "auto"
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        strategy=strategy,
        devices=gpu_count,
    )

    ckpt_path = None
    resume_ckpt = cfg.get("resume_from_checkpoint", None)

    if resume_ckpt:
        ckpt_path = resume_ckpt
        print(f"Resuming from explicit checkpoint: {ckpt_path}")
    else:
        auto_last = ckpt_dir / "last.ckpt"
        if auto_last.exists():
            ckpt_path = str(auto_last)
            print(f"Auto-resuming from: {ckpt_path}")
        else:
            print("No checkpoint found — starting fresh.")

    if (
        hasattr(logger, "experiment")
        and hasattr(logger.experiment, "id")
        and cfg.logger.mode != "disabled"
    ):
        write_wandb_id(logger.experiment.id, wandb_id_file)

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
