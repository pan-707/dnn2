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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

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


class VGG16FC7Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            vgg = models.vgg16(weights=weights)
        except AttributeError:
            vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc7 = nn.Sequential(*list(vgg.classifier.children())[:5])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc7(x)


def parse_args():
    parser = argparse.ArgumentParser(description="PCA on 4096-dim VGG16 features.")
    parser.add_argument("--data-root", type=Path, default=Path("/export/space0/pan-p/data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--feature-limit", type=int, default=500)
    parser.add_argument("--cluster-limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_loader(args):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    raw_transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(
        root=str(args.data_root),
        train=True,
        transform=transform,
        download=args.download,
    )
    raw_dataset = datasets.CIFAR10(
        root=str(args.data_root),
        train=True,
        transform=raw_transform,
        download=False,
    )
    limit = min(args.feature_limit, len(dataset))
    indices = list(range(limit))
    return (
        DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        ),
        Subset(raw_dataset, indices),
    )


def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            out = model(images).cpu().numpy().astype(np.float32)
            features.append(out)
            labels.append(target.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def reduce_features(features):
    reducers = {
        "4096dim": None,
        "pca95": PCA(n_components=0.95, svd_solver="full"),
        "pca90": PCA(n_components=0.90, svd_solver="full"),
        "pca128": PCA(n_components=128, svd_solver="randomized", random_state=7),
    }
    reduced = {}
    fitted = {}
    for name, reducer in reducers.items():
        if reducer is None:
            reduced[name] = features
        else:
            reduced[name] = reducer.fit_transform(features)
            fitted[name] = reducer
    return reduced, fitted


def cluster_summary(labels, clusters):
    rows = []
    for cluster_id in range(clusters.max() + 1):
        idx = np.where(clusters == cluster_id)[0]
        counts = np.bincount(labels[idx], minlength=len(CLASS_NAMES))
        top = counts.argsort()[::-1][:3]
        top_classes = " / ".join(f"{CLASS_NAMES[i]}:{counts[i]}" for i in top if counts[i] > 0)
        rows.append([cluster_id, len(idx), top_classes])
    return rows


def save_cluster_csv(results, labels):
    with (DATA_DIR / "pca_cluster_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["representation", "k", "cluster", "size", "top_classes"])
        for rep_name, k_to_clusters in results.items():
            for k, clusters in k_to_clusters.items():
                for cluster_id, size, top_classes in cluster_summary(labels, clusters):
                    writer.writerow([rep_name, k, cluster_id, size, top_classes])


def save_pca_csv(fitted, reduced):
    with (DATA_DIR / "pca_dimensions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["representation", "dimension", "explained_variance_ratio_sum"])
        writer.writerow(["4096dim", reduced["4096dim"].shape[1], "1.000000"])
        for name, reducer in fitted.items():
            writer.writerow(
                [
                    name,
                    reduced[name].shape[1],
                    f"{float(np.sum(reducer.explained_variance_ratio_)):.6f}",
                ]
            )


def save_cluster_sheet(raw_dataset, labels, clusters, rep_name, k, samples_per_cluster=8):
    fig, axes = plt.subplots(k, samples_per_cluster, figsize=(samples_per_cluster * 1.35, k * 1.25))
    if k == 1:
        axes = np.expand_dims(axes, axis=0)

    for cluster_id in range(k):
        idx = np.where(clusters == cluster_id)[0][:samples_per_cluster]
        for col in range(samples_per_cluster):
            ax = axes[cluster_id, col]
            ax.axis("off")
            if col < len(idx):
                image, label = raw_dataset[int(idx[col])]
                ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
                ax.set_title(CLASS_NAMES[int(label)], fontsize=7)
        axes[cluster_id, 0].set_ylabel(f"C{cluster_id}", fontsize=8)

    fig.suptitle(f"{rep_name}, k={k}", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"clusters_{rep_name}_k{k}.png", dpi=180)
    plt.close(fig)


def save_dimension_plot(fitted, reduced):
    names = ["4096dim", "pca95", "pca90", "pca128"]
    dims = [reduced[name].shape[1] for name in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, dims, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_ylabel("dimension")
    ax.set_title("Feature dimension after PCA")
    for i, dim in enumerate(dims):
        ax.text(i, dim, str(dim), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_dimensions.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.feature_limit < 129:
        raise ValueError("--feature-limit must be at least 129 to compute PCA 128 dimensions.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    loader, raw_dataset = make_loader(args)
    model = VGG16FC7Extractor().to(device)
    features, labels = extract_features(model, loader, device)
    print(f"features: {features.shape}")

    reduced, fitted = reduce_features(features)
    save_pca_csv(fitted, reduced)
    save_dimension_plot(fitted, reduced)

    cluster_limit = min(args.cluster_limit, len(labels))
    cluster_labels = labels[:cluster_limit]
    results = {}
    for rep_name, rep_features in reduced.items():
        results[rep_name] = {}
        x = rep_features[:cluster_limit]
        for k in [5, 10]:
            kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
            clusters = kmeans.fit_predict(x)
            results[rep_name][k] = clusters
            save_cluster_sheet(raw_dataset, cluster_labels, clusters, rep_name, k)

    save_cluster_csv(results, cluster_labels)
    np.savez_compressed(DATA_DIR / "vgg16_fc7_features.npz", features=features, labels=labels)

    print(f"saved: {DATA_DIR / 'pca_dimensions.csv'}")
    print(f"saved: {DATA_DIR / 'pca_cluster_summary.csv'}")
    print(f"saved: {FIG_DIR / 'pca_dimensions.png'}")


if __name__ == "__main__":
    main()
