import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric
from homework.models import Detector, save_model


def train(
    exp_dir: str = "logs",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    weight_decay: float = 1e-4,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = Detector().to(device)

    train_data = load_data(
        "drive_data/train",
        transform_pipeline="default",
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )

    val_data = load_data(
        "drive_data/val",
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_iou = 0.0
    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_metric = DetectionMetric()
        train_metric.reset()

        for batch in train_data:
            img = torch.tensor(batch["image"]).to(device)
            track = torch.tensor(batch["track"]).long().to(device)
            depth = torch.tensor(batch["depth"]).to(device)

            optimizer.zero_grad()

            logits, depth_pred = model(img)

            seg_loss = seg_loss_fn(logits, track)
            depth_loss = depth_loss_fn(depth_pred, depth)

            loss = seg_loss + 0.5 * depth_loss

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_metric.add(preds.cpu(), track.cpu(), depth_pred.cpu(), depth.cpu())

            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        # validation
        model.eval()
        val_metric = DetectionMetric()
        val_metric.reset()

        with torch.inference_mode():
            for batch in val_data:
                img = torch.tensor(batch["image"]).to(device)
                track = torch.tensor(batch["track"]).long().to(device)
                depth = torch.tensor(batch["depth"]).to(device)

                logits, depth_pred = model(img)

                preds = logits.argmax(dim=1)
                val_metric.add(preds.cpu(), track.cpu(), depth_pred.cpu(), depth.cpu())

        train_stats = train_metric.compute()
        val_stats = val_metric.compute()

        logger.add_scalar("val_iou", val_stats["iou"], epoch)

        print(
            f"Epoch {epoch+1:02d}/{num_epoch} | "
            f"train_iou={train_stats['iou']:.4f} | "
            f"val_iou={val_stats['iou']:.4f} | "
            f"val_depth_err={val_stats['abs_depth_error']:.4f}"
        )

        # save best model
        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            save_model(model)
            torch.save(model.state_dict(), log_dir / "detector.th")

    print(f"Best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)

    train(**vars(parser.parse_args()))