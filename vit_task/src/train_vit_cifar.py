import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TinyVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        num_tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small Vision Transformer on CIFAR-10.")
    parser.add_argument("--data-root", type=Path, default=Path("/export/space0/pan-p/data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--val-limit", type=int, default=2000)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_loaders(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    raw_transform = transforms.ToTensor()

    train_set = datasets.CIFAR10(
        root=str(args.data_root),
        train=True,
        transform=train_transform,
        download=args.download,
    )
    val_set = datasets.CIFAR10(
        root=str(args.data_root),
        train=False,
        transform=eval_transform,
        download=args.download,
    )
    raw_val_set = datasets.CIFAR10(
        root=str(args.data_root),
        train=False,
        transform=raw_transform,
        download=False,
    )

    if args.train_limit > 0:
        train_set = Subset(train_set, range(min(args.train_limit, len(train_set))))
    if args.val_limit > 0:
        val_indices = range(min(args.val_limit, len(val_set)))
        val_set = Subset(val_set, val_indices)
        raw_val_set = Subset(raw_val_set, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, raw_val_set


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_acc += float(accuracy(logits, labels).item()) * batch_size
            seen += batch_size
    return total_loss / seen, total_acc / seen


def save_training_curve(history):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], marker="o", label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.png", dpi=180)
    plt.close(fig)


def save_predictions(model, eval_loader, raw_val_set, device):
    model.eval()
    images, labels = next(iter(eval_loader))
    images = images[:16].to(device)
    labels = labels[:16]
    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        raw_image, _ = raw_val_set[i]
        ax.imshow(np.transpose(raw_image.numpy(), (1, 2, 0)))
        color = "green" if int(preds[i]) == int(labels[i]) else "red"
        ax.set_title(
            f"pred: {CLASS_NAMES[int(preds[i])]}\ntrue: {CLASS_NAMES[int(labels[i])]}",
            fontsize=8,
            color=color,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predictions.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"data_root: {args.data_root}")

    train_loader, val_loader, raw_val_set = make_loaders(args)
    model = TinyVisionTransformer(
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            running_acc += float(accuracy(logits, labels).item()) * batch_size
            seen += batch_size

        scheduler.step()
        train_loss = running_loss / seen
        train_acc = running_acc / seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), CKPT_DIR / "tiny_vit_cifar10.pt")
    with (DATA_DIR / "training_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)

    save_training_curve(history)
    save_predictions(model, val_loader, raw_val_set, device)
    print(f"saved: {DATA_DIR / 'training_metrics.csv'}")
    print(f"saved: {FIG_DIR / 'training_curve.png'}")
    print(f"saved: {FIG_DIR / 'predictions.png'}")


if __name__ == "__main__":
    main()
