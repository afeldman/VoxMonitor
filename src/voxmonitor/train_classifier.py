import os
import yaml
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from voxmonitor.data import SoundwelDataset, DynamicAugmentor, train_val_split
from voxmonitor.model import MultiTaskMobileNet


def get_device(cfg_device: str) -> torch.device:
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(cfg_path: str = "config/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    labels_cfg = cfg["labels"]
    mel_cfg = cfg["mel"]
    aug_cfg = cfg["augment"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)

    augmentor = DynamicAugmentor(
        factory_path=aug_cfg.get("factory_path", ""), params=aug_cfg.get("params", {})
    ) if aug_cfg.get("enabled", False) else None

    # Build full dataset to collect label mappings
    full_ds = SoundwelDataset(
        root_dir=data_cfg["root_dir"],
        key_xlsx=data_cfg["key_xlsx"],
        audio_col_candidates=data_cfg["audio_column_candidates"],
        label_columns=labels_cfg["columns"],
        sample_rate=data_cfg["sample_rate"],
        mono=data_cfg["mono"],
        mel_cfg=mel_cfg,
        augmentor=augmentor,
    )

    # Derive class counts from label mappings
    num_classes: Dict[str, int] = {t: len(full_ds.label_mappings[t]) for t in labels_cfg["columns"]}

    # Train/val split
    train_idx, val_idx = train_val_split(len(full_ds), data_cfg["val_fraction"], seed=train_cfg["seed"])

    train_ds = SoundwelDataset(
        root_dir=data_cfg["root_dir"],
        key_xlsx=data_cfg["key_xlsx"],
        audio_col_candidates=data_cfg["audio_column_candidates"],
        label_columns=labels_cfg["columns"],
        sample_rate=data_cfg["sample_rate"],
        mono=data_cfg["mono"],
        mel_cfg=mel_cfg,
        augmentor=augmentor,
        indices=train_idx,
    )

    val_ds = SoundwelDataset(
        root_dir=data_cfg["root_dir"],
        key_xlsx=data_cfg["key_xlsx"],
        audio_col_candidates=data_cfg["audio_column_candidates"],
        label_columns=labels_cfg["columns"],
        sample_rate=data_cfg["sample_rate"],
        mono=data_cfg["mono"],
        mel_cfg=mel_cfg,
        augmentor=None,  # no augmentation on val
        indices=val_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"], drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"], drop_last=False)

    device = get_device(train_cfg.get("device", "auto"))
    model = MultiTaskMobileNet(num_classes=num_classes, pretrained=model_cfg.get("pretrained", True))
    model.to(device)

    if model_cfg.get("warmup_freeze_epochs", 0) > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False

    criterion = {t: nn.CrossEntropyLoss() for t in labels_cfg["columns"]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    best_val_loss = float("inf")
    for epoch in range(train_cfg["max_epochs"]):
        # Unfreeze after warmup
        if epoch == model_cfg.get("warmup_freeze_epochs", 0):
            for p in model.backbone.parameters():
                p.requires_grad = True

        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['max_epochs']} [train]"):
            xb = xb.to(device)
            logits = model(xb)
            loss = 0.0
            for t in labels_cfg["columns"]:
                yt = torch.tensor([yy[t] for yy in yb], dtype=torch.long, device=device)
                loss = loss + criterion[t](logits[t], yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1}/{train_cfg['max_epochs']} [val]"):
                xb = xb.to(device)
                logits = model(xb)
                loss = 0.0
                for t in labels_cfg["columns"]:
                    yt = torch.tensor([yy[t] for yy in yb], dtype=torch.long, device=device)
                    loss = loss + criterion[t](logits[t], yt)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        # Save checkpoint
        ckpt_path = os.path.join(train_cfg["checkpoint_dir"], f"epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "label_mappings": full_ds.label_mappings,
        }, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "label_mappings": full_ds.label_mappings,
            }, os.path.join(train_cfg["checkpoint_dir"], "best.pt"))


if __name__ == "__main__":
    raise NotImplementedError("VoxMonitor nutzt aptt-Training. Bitte aptt verwenden.")
