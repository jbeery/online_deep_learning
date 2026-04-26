"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse

import torch
import torch.nn as nn

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model


def train(
    model_name: str = "mlp_planner",
    train_path: str = "drive_data/train",
    val_path: str = "drive_data/val",
    num_epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_pipeline = "default"
    if model_name in ["mlp_planner", "transformer_planner"]:
        transform_pipeline = "state_only"

    train_data = load_data(
        train_path,
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        shuffle=True,
    )
    val_data = load_data(
        val_path,
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        shuffle=False,
    )

    model = load_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    best_val_score = float("inf")
    best_state_dict = None

    for epoch in range(num_epochs):
        model.train()

        train_metric = PlannerMetric()
        train_loss = 0.0

        for batch in train_data:
            batch = {k: v.to(device) for k, v in batch.items()}

            preds = model(**batch)
            loss = loss_fn(preds, batch["waypoints"])
            loss = loss * batch["waypoints_mask"][..., None]
            loss = loss.sum() / batch["waypoints_mask"].sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_metric.add(preds, batch["waypoints"], batch["waypoints_mask"])

        model.eval()

        val_metric = PlannerMetric()

        with torch.inference_mode():
            for batch in val_data:
                batch = {k: v.to(device) for k, v in batch.items()}

                preds = model(**batch)
                val_metric.add(preds, batch["waypoints"], batch["waypoints_mask"])

        train_results = train_metric.compute()
        val_results = val_metric.compute()

        val_score = val_results["longitudinal_error"] + val_results["lateral_error"]

        if val_score < best_val_score:
            best_val_score = val_score
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"epoch {epoch + 1:02d} / {num_epochs:02d} "
            f"loss={train_loss / len(train_data):.4f} "
            f"train_long={train_results['longitudinal_error']:.4f} "
            f"train_lat={train_results['lateral_error']:.4f} "
            f"val_long={val_results['longitudinal_error']:.4f} "
            f"val_lat={val_results['lateral_error']:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    save_model(model)