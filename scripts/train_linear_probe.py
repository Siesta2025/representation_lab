import torch
import numpy as np
import random
import torch.nn as nn
import argparse
import csv
import json
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Subset
from representation_lab.linear_probe import LinearProbe
from representation_lab.models import SmallResNet
from representation_lab.classification_data import get_train_val_dataset, get_test_dataset, load_data
from representation_lab.training import train_linear_probe_one_epoch, evaluate_classifier
from representation_lab.checkpoints import (
    load_encoder_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from representation_lab.logging_utils import setup_logger

def initialize_linear_probe(encoder, num_classes=10):
    prober = LinearProbe(encoder, num_classes=num_classes)
    for param in prober.frozen_encoder.parameters():
        assert param.requires_grad is False
    for param in prober.classifier.parameters():
        assert param.requires_grad is True
    return prober

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
        default="training",
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
        "--num_classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        choices=["random", "simclr", "supervised"],
        default="simclr",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    num_workers = args.num_workers
    seed = args.seed
    lr = args.lr
    momentum = args.momentum
    num_epochs = args.num_epochs
    run_name = args.run_name
    resume = args.resume
    milestones = args.milestones
    gamma = args.gamma
    num_classes = args.num_classes
    encoder_name = args.encoder_name

    path = Path("./outputs") / run_name

    if resume is None:
        assert not path.exists(), f"Run name '{run_name}' already exists. Please choose a different run name."

    elif resume == "last":
        assert path.exists(), f"Run name '{run_name}' does not exist. Please choose a valid run name to resume training."
    
    else:
        assert path.exists(), f"Run name '{run_name}' does not exist. Please choose a valid run name to resume training."
    
    set_seed(seed)

    generator = torch.Generator().manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    train_val_dataset = get_train_val_dataset("./data", transform)

    test_dataset = get_test_dataset("./data", transform)

    indices = torch.randperm(
        50000,
        generator=generator,
    )
    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_dataset = Subset(
        train_val_dataset,
        train_indices,
    )
    val_dataset = Subset(
        train_val_dataset,
        val_indices,
    )

    train_loader = load_data(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_loader = load_data(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    test_loader = load_data(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    encoder = SmallResNet()
    if encoder_name == "simclr":
        load_encoder_checkpoint(encoder, "./outputs/simclr_30/best.pt")
    elif encoder_name == "supervised":
        load_encoder_checkpoint(encoder, "./outputs/supervised_30/best.pt")
    model = initialize_linear_probe(encoder, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.classifier.parameters(),
        lr=lr,
        momentum=momentum,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma,
    )

    last_checkpoint_path = path / "last.pt"
    best_checkpoint_path = path / "best.pt"

    if resume is None:
        path.mkdir(parents=True)

    logger = setup_logger(name=__name__, log_path=path / "train.log")
    logger.info(
        f"run_started run_name={run_name} device={device} "
        f"resume={resume} encoder={encoder_name}"
    )

    if resume is None:
        start_epoch = 0
        best_val_accuracy = 0.0

        config = vars(args).copy()
        config["device"] = str(device)
        config["dataset"] = "CIFAR10"
        config["criterion"] = "CrossEntropyLoss"
        config["optimizer"] = "SGD"
        config["scheduler"] = "MultiStepLR"

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        logger.info(f"config_saved path={path / 'config.json'}")

    elif resume == "last":
        start_epoch, best_val_accuracy = load_training_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        logger.info(
            f"training_resumed checkpoint={last_checkpoint_path} "
            f"start_epoch={start_epoch} "
            f"best_val_accuracy={best_val_accuracy:.4f}"
        )

    else:
        start_epoch, best_val_accuracy = load_training_checkpoint(
            path=best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        logger.info(
            f"training_resumed checkpoint={best_checkpoint_path} "
            f"start_epoch={start_epoch} "
            f"best_val_accuracy={best_val_accuracy:.4f}"
        )

    metric_path = path / "metrics.csv"
    metric_fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]
    if not metric_path.exists():
        with open(metric_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_accuracy = train_linear_probe_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate_classifier(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"epoch_complete epoch={epoch + 1} "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )
        scheduler.step()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_training_checkpoint(
                path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_val_accuracy,
            )
            logger.info(
                f"best_checkpoint_saved epoch={epoch + 1} "
                f"val_accuracy={best_val_accuracy:.4f} "
                f"path={best_checkpoint_path}"
            )

        save_training_checkpoint(
            path=last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_val_accuracy,
        )
        logger.info(
            f"last_checkpoint_saved epoch={epoch + 1} "
            f"path={last_checkpoint_path}"
        )

        metrics = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
        }
        with open(metric_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writerow(metrics)

    load_training_checkpoint(
        path=best_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    logger.info(f"best_checkpoint_loaded path={best_checkpoint_path}")

    test_loss, test_accuracy = evaluate_classifier(model, test_loader, criterion, device)
    logger.info(
        f"run_complete test_loss={test_loss:.4f} "
        f"test_accuracy={test_accuracy:.4f}"
    )
