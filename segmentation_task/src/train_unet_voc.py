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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_COLORS = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small UNet on PASCAL VOC segmentation masks.")
    parser.add_argument("--voc-root", type=Path, default=Path("/export/data/dataset/VOCdevkit/VOC2012"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=800)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_split(voc_root, split):
    split_file = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if split_file.exists():
        return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

    mask_dir = voc_root / "SegmentationClass"
    ids = sorted(path.stem for path in mask_dir.glob("*.png"))
    cut = int(len(ids) * 0.8)
    return ids[:cut] if split == "train" else ids[cut:]


class VOCDataset(Dataset):
    def __init__(self, voc_root, split, image_size=128, limit=0):
        self.voc_root = Path(voc_root)
        self.image_dir = self.voc_root / "JPEGImages"
        self.mask_dir = self.voc_root / "SegmentationClass"
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"VOC root must contain JPEGImages and SegmentationClass: {self.voc_root}"
            )

        ids = read_split(self.voc_root, split)
        ids = [
            image_id
            for image_id in ids
            if (self.image_dir / f"{image_id}.jpg").exists()
            and (self.mask_dir / f"{image_id}.png").exists()
        ]
        if limit > 0:
            ids = ids[:limit]
        if not ids:
            raise RuntimeError(f"No VOC segmentation samples found for split={split}.")
        self.ids = ids
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{image_id}.png")
        image = self.image_transform(image)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.bridge(self.pool2(x2))
        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.out(x)


def pixel_accuracy(logits, target):
    pred = logits.argmax(dim=1)
    valid = target != 255
    correct = (pred[valid] == target[valid]).sum().float()
    total = valid.sum().float().clamp_min(1)
    return correct / total


def mean_iou(logits, target, num_classes=21):
    pred = logits.argmax(dim=1)
    valid = target != 255
    ious = []
    for cls in range(num_classes):
        pred_c = (pred == cls) & valid
        target_c = (target == cls) & valid
        union = pred_c | target_c
        if union.any():
            intersection = pred_c & target_c
            ious.append(intersection.sum().float() / union.sum().float())
    if not ious:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(ious).mean()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    seen = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_acc += float(pixel_accuracy(logits, masks).item()) * batch_size
            total_miou += float(mean_iou(logits, masks).item()) * batch_size
            seen += batch_size
    return total_loss / seen, total_acc / seen, total_miou / seen


def colorize_mask(mask):
    mask = np.asarray(mask)
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(VOC_COLORS))
    colored[valid] = VOC_COLORS[mask[valid]]
    return colored


def save_training_curve(history):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["val_pixel_acc"] for row in history], marker="o", label="pixel acc")
    axes[1].plot(epochs, [row["val_miou"] for row in history], marker="o", label="mIoU")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.png", dpi=180)
    plt.close(fig)


def save_predictions(model, loader, device):
    model.eval()
    images, masks = next(iter(loader))
    images = images[:6].to(device)
    masks = masks[:6].numpy()
    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu().numpy()
    images = images.cpu().numpy()

    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    for i in range(6):
        image = np.transpose(images[i], (1, 2, 0))
        gt = colorize_mask(masks[i])
        pred = colorize_mask(preds[i])
        overlay = (0.55 * image + 0.45 * (pred.astype(np.float32) / 255.0)).clip(0, 1)

        axes[0, i].imshow(image)
        axes[1, i].imshow(gt)
        axes[2, i].imshow(pred)
        axes[3, i].imshow(overlay)
        for row in range(4):
            axes[row, i].axis("off")

    axes[0, 0].set_ylabel("image")
    axes[1, 0].set_ylabel("gt")
    axes[2, 0].set_ylabel("pred")
    axes[3, 0].set_ylabel("overlay")
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
    print(f"voc_root: {args.voc_root}")

    train_dataset = VOCDataset(args.voc_root, "train", args.image_size, args.train_limit)
    val_dataset = VOCDataset(args.voc_root, "val", args.image_size, args.val_limit)
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNetSmall(num_classes=len(VOC_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / seen
        val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pixel_acc": val_acc,
            "val_miou": val_miou,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_pixel_acc={val_acc:.4f} val_miou={val_miou:.4f}"
        )

    torch.save(model.state_dict(), CKPT_DIR / "unet_voc.pt")
    with (DATA_DIR / "training_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_pixel_acc", "val_miou"],
        )
        writer.writeheader()
        writer.writerows(history)

    save_training_curve(history)
    save_predictions(model, val_loader, device)
    print(f"saved: {DATA_DIR / 'training_metrics.csv'}")
    print(f"saved: {FIG_DIR / 'training_curve.png'}")
    print(f"saved: {FIG_DIR / 'predictions.png'}")


if __name__ == "__main__":
    main()
