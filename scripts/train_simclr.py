import argparse
import csv
import json
import random
import wandb
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from representation_lab.checkpoints import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from representation_lab.logging_utils import setup_logger
from representation_lab.models import SimCLRModel, ProjectionHead, SmallResNet
from representation_lab.simclr_data import get_train_val_dataset
from representation_lab.losses import NTXent
from representation_lab.training import train_simclr_one_epoch, evaluate_simclr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="simclr",
    )
    parser.add_argument(
        "--resume",
        type=str,
        choices=["last", "best"],
        default=None,
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[20, 25],
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="disabled",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="representation_lab",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resume = args.resume
    run_name = args.run_name
    seed = args.seed
    batch_size = args.batch_size
    num_workers = args.num_workers
    tau = args.tau
    lr = args.lr
    milestones = args.milestones
    gamma = args.gamma
    momentum = args.momentum
    num_epochs = args.num_epochs
    wandb_mode = args.wandb_mode
    wandb_project = args.wandb_project
    wandb_id = args.wandb_id

    wandb_dir = Path("./outputs").resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)

    wandb_resume = (
        "must"
        if resume == "last"
        and wandb_id is not None
        and wandb_mode == "online"
        else None
    )

    path = Path("./outputs") / run_name

    if resume is None:
        assert not path.exists(), (
            f"Run '{run_name}' already exists."
        )
    else:
        assert path.exists(), (
            f"Run '{run_name}' does not exist."
        )

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------
    # Data
    # -------------------------

    generator = torch.Generator().manual_seed(
        seed
    )

    full_dataset = get_train_val_dataset(
        "./data"
    )

    indices = torch.randperm(
        len(full_dataset),
        generator=generator,
    )

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_dataset = Subset(
        full_dataset,
        train_indices,
    )

    val_dataset = Subset(
        full_dataset,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # -------------------------
    # Model
    # -------------------------

    encoder = SmallResNet()

    projector = ProjectionHead(
        in_dim=256,
        out_dim=128,
    )

    model = SimCLRModel(
        encoder,
        projector,
    ).to(device)

    criterion = NTXent(
        tau=tau
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma,
    )

    # -------------------------
    # Run state
    # -------------------------

    last_checkpoint_path = path / "last.pt"
    best_checkpoint_path = path / "best.pt"
    metric_path = path / "metrics.csv"

    metric_fields = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
    ]

    if resume is None:
        path.mkdir(parents=True)

    logger = setup_logger(name=__name__, log_path=path / "train.log")
    logger.info(
        f"run_started run_name={run_name} device={device} "
        f"resume={resume}"
    )

    config = vars(args).copy()
    config["device"] = str(device)
    config["dataset"] = "CIFAR10"
    config["model"] = "SmallResNet+ProjectionHead"
    config["projection_dim"] = 128
    config["criterion"] = "NTXent"
    config["optimizer"] = "SGD"
    config["scheduler"] = "MultiStepLR"
    config["train_size"] = 45000
    config["val_size"] = 5000

    if resume is None:
        start_epoch = 0
        best_val_loss = float("inf")

        with open(
            path / "config.json",
            "w",
        ) as f:
            json.dump(
                config,
                f,
                indent=4,
            )
        logger.info(f"config_saved path={path / 'config.json'}")

    else:
        checkpoint_path = (
            last_checkpoint_path
            if resume == "last"
            else best_checkpoint_path
        )

        assert checkpoint_path.exists(), (
            f"Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

        start_epoch, best_val_loss = load_training_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        logger.info(
            f"training_resumed checkpoint={checkpoint_path} "
            f"start_epoch={start_epoch} "
            f"best_val_loss={best_val_loss:.4f}"
        )

    if not metric_path.exists():
        with open(
            metric_path,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()
        logger.info(f"metrics_header_written path={metric_path}")

    run = wandb.init(
        id=wandb_id,
        project=wandb_project,
        name=run_name,
        config=config,
        mode=wandb_mode,
        dir=str(wandb_dir),
        resume=wandb_resume,
        allow_val_change=resume is not None,
    )
    logger.info(
        f"wandb_run_initialized run_id={run.id} "
        f"resumed={run.resumed} "
        f"run_name={run_name} "
        f"wandb_mode={wandb_mode} "
        f"wandb_project={wandb_project}"
    )

    # -------------------------
    # Training
    # -------------------------

    for epoch in range(
        start_epoch,
        num_epochs,
    ):
        train_loss = train_simclr_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss = evaluate_simclr(
            model,
            val_loader,
            criterion,
            device,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        logger.info(
            f"epoch_complete epoch={epoch + 1} "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_training_checkpoint(
                path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_val_loss,
            )

            logger.info(
                f"best_checkpoint_saved epoch={epoch + 1} "
                f"val_loss={best_val_loss:.4f} "
                f"path={best_checkpoint_path}"
            )

        save_training_checkpoint(
            path=last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_val_loss,
        )

        logger.info(
            f"last_checkpoint_saved epoch={epoch + 1} "
            f"path={last_checkpoint_path}"
        )

        metrics = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        with open(
            metric_path,
            "a",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writerow(metrics)
        run.log(metrics)

    logger.info(
        f"run_complete best_val_loss={best_val_loss:.4f}"
    )
    run.summary["best_val_loss"] = best_val_loss
    run.finish()
    logger.info(
        f"wandb_run_finished run_id={run.id} "
        f"run_name={run_name} "
        f"wandb_mode={wandb_mode} "
        f"wandb_project={wandb_project}"
    )
