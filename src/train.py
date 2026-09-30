import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import FabricDataset
from transforms import train_transform, val_transform
from model import SmallViT

import config


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    config.DEVICE
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# DATASETS
# ============================================================

train_dataset = FabricDataset(
    config.TRAIN_IMAGE_DIR,
    config.TRAIN_XML_DIR,
    transform=train_transform
)

val_dataset = FabricDataset(
    config.VAL_IMAGE_DIR,
    config.VAL_XML_DIR,
    transform=val_transform
)


print(
    f"Train samples: {len(train_dataset)}"
)

print(
    f"Val samples: {len(val_dataset)}"
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(
        config.NUM_WORKERS > 0
    )
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(
        config.NUM_WORKERS > 0
    )
)


# ============================================================
# MODEL
# ============================================================

model = SmallViT(
    image_size=config.IMAGE_SIZE,
    embed_dim=config.EMBED_DIM,
    depth=config.DEPTH,
    num_heads=config.NUM_HEADS,
    mlp_ratio=config.MLP_RATIO,
    dropout=config.DROPOUT,
    num_pattern_classes=config.NUM_PATTERN_CLASSES,
    num_defect_classes=config.NUM_DEFECT_CLASSES
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

pattern_criterion = nn.CrossEntropyLoss()

defect_criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.EPOCHS
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")


for epoch in range(config.EPOCHS):

    model.train()

    running_loss = 0.0
    pattern_correct = 0
    defect_correct = 0
    total = 0

    for batch in train_loader:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        pattern_targets = batch["pattern"].to(
            device,
            non_blocking=True
        )

        defect_targets = batch["defect"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        pattern_loss = pattern_criterion(
            outputs["pattern"],
            pattern_targets
        )

        defect_loss = defect_criterion(
            outputs["defect"],
            defect_targets
        )

        loss = (
            config.PATTERN_LOSS_WEIGHT * pattern_loss
            +
            config.DEFECT_LOSS_WEIGHT * defect_loss
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        pattern_pred = (
            outputs["pattern"].argmax(dim=1)
        )

        defect_pred = (
            outputs["defect"].argmax(dim=1)
        )

        pattern_correct += (
            pattern_pred == pattern_targets
        ).sum().item()

        defect_correct += (
            defect_pred == defect_targets
        ).sum().item()

        total += images.size(0)

    scheduler.step()

    train_loss = (
        running_loss / total
    )

    train_pattern_acc = (
        pattern_correct / total
    )

    train_defect_acc = (
        defect_correct / total
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_pattern_correct = 0
    val_defect_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(device)

            pattern_targets = batch[
                "pattern"
            ].to(device)

            defect_targets = batch[
                "defect"
            ].to(device)

            outputs = model(images)

            pattern_loss = pattern_criterion(
                outputs["pattern"],
                pattern_targets
            )

            defect_loss = defect_criterion(
                outputs["defect"],
                defect_targets
            )

            loss = (
                config.PATTERN_LOSS_WEIGHT * pattern_loss
                +
                config.DEFECT_LOSS_WEIGHT * defect_loss
            )

            val_loss += (
                loss.item() * images.size(0)
            )

            pattern_pred = (
                outputs["pattern"].argmax(dim=1)
            )

            defect_pred = (
                outputs["defect"].argmax(dim=1)
            )

            val_pattern_correct += (
                pattern_pred == pattern_targets
            ).sum().item()

            val_defect_correct += (
                defect_pred == defect_targets
            ).sum().item()

            val_total += images.size(0)

    val_loss /= val_total

    val_pattern_acc = (
        val_pattern_correct / val_total
    )

    val_defect_acc = (
        val_defect_correct / val_total
    )

    print(
        f"\nEpoch [{epoch + 1}/{config.EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Pattern Acc: {train_pattern_acc:.4f} | "
        f"Defect Acc: {train_defect_acc:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Pattern Acc: {val_pattern_acc:.4f} | "
        f"Defect Acc: {val_defect_acc:.4f}"
    )

    print(
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint_path = (
            config.CHECKPOINT_DIR
            / "best_model.pth"
        )

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, checkpoint_path)

        print(
            f"Saved best model → {checkpoint_path}"
        )


print("\nTraining complete.")