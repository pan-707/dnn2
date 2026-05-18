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
from sklearn.cluster import KMeans
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


class ConvAutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Linear(128 * 4 * 4, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_cnn(x)
        z = torch.flatten(z, start_dim=1)
        return self.encoder_fc(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 4, 4)
        return self.decoder_cnn(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 convolutional autoencoder.")
    parser.add_argument("--data-root", type=Path, default=Path("/export/data/dataset"))
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if needed.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(
        root=str(args.data_root),
        train=True,
        transform=transform,
        download=args.download,
    )
    test_set = datasets.CIFAR10(
        root=str(args.data_root),
        train=False,
        transform=transform,
        download=args.download,
    )

    if args.train_limit > 0:
        train_set = Subset(train_set, range(min(args.train_limit, len(train_set))))
    if args.eval_limit > 0:
        test_set = Subset(test_set, range(min(args.eval_limit, len(test_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader


def save_reconstructions(model: ConvAutoEncoder, loader: DataLoader, device: torch.device) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images = images[:10].to(device)
    labels = labels[:10]
    with torch.no_grad():
        recon = model(images).cpu()
    images = images.cpu()

    fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
    for i in range(10):
        axes[0, i].imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        axes[0, i].set_title(CLASS_NAMES[int(labels[i])], fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(np.transpose(recon[i].numpy(), (1, 2, 0)))
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=10)
    axes[1, 0].set_ylabel("recon", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reconstructions.png", dpi=180)
    plt.close(fig)


def collect_features(
    model: ConvAutoEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_features = []
    all_images = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.encode(images).cpu().numpy()
            all_features.append(features)
            all_images.append(images.cpu().numpy())
            all_labels.append(labels.numpy())
    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_images, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def save_cluster_sheet(
    images: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    k: int,
    samples_per_cluster: int = 8,
) -> None:
    fig, axes = plt.subplots(k, samples_per_cluster, figsize=(samples_per_cluster * 1.5, k * 1.35))
    if k == 1:
        axes = np.expand_dims(axes, 0)

    for cluster_id in range(k):
        idx = np.where(clusters == cluster_id)[0][:samples_per_cluster]
        for col in range(samples_per_cluster):
            ax = axes[cluster_id, col]
            ax.axis("off")
            if col < len(idx):
                image = np.transpose(images[idx[col]], (1, 2, 0))
                ax.imshow(image)
                ax.set_title(CLASS_NAMES[int(labels[idx[col]])], fontsize=7)
        axes[cluster_id, 0].set_ylabel(f"C{cluster_id}", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"kmeans_k{k}.png", dpi=180)
    plt.close(fig)


def save_cluster_summary(labels: np.ndarray, clusters_by_k: dict[int, np.ndarray]) -> None:
    with (DATA_DIR / "cluster_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "cluster", "size", "top_classes"])
        for k, clusters in clusters_by_k.items():
            for cluster_id in range(k):
                idx = np.where(clusters == cluster_id)[0]
                counts = np.bincount(labels[idx], minlength=len(CLASS_NAMES))
                top = counts.argsort()[::-1][:3]
                top_classes = " / ".join(f"{CLASS_NAMES[i]}:{counts[i]}" for i in top if counts[i] > 0)
                writer.writerow([k, cluster_id, len(idx), top_classes])


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    FIG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, eval_loader = make_loaders(args)
    model = ConvAutoEncoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, _ in train_loader:
            images = images.to(device)
            recon = model(images)
            loss = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
        train_loss = running_loss / seen
        history.append(train_loss)
        print(f"epoch {epoch:02d}/{args.epochs}: train_mse={train_loss:.6f}")

    torch.save(model.state_dict(), CKPT_DIR / "autoencoder.pt")

    with (DATA_DIR / "training_loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse"])
        for epoch, loss in enumerate(history, start=1):
            writer.writerow([epoch, f"{loss:.8f}"])

    save_reconstructions(model, eval_loader, device)
    features, images, labels = collect_features(model, eval_loader, device)

    clusters_by_k = {}
    for k in [5, 10]:
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        clusters = kmeans.fit_predict(features)
        clusters_by_k[k] = clusters
        save_cluster_sheet(images, labels, clusters, k)

    save_cluster_summary(labels, clusters_by_k)
    print(f"saved: {FIG_DIR / 'reconstructions.png'}")
    print(f"saved: {FIG_DIR / 'kmeans_k5.png'}")
    print(f"saved: {FIG_DIR / 'kmeans_k10.png'}")
    print(f"saved: {DATA_DIR / 'training_loss.csv'}")
    print(f"saved: {DATA_DIR / 'cluster_summary.csv'}")


if __name__ == "__main__":
    main()
