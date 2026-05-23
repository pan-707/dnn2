import argparse
import csv
import json
import os
import random
import re
import textwrap
from collections import Counter
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
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

PAD = "<pad>"
UNK = "<unk>"
EOS = "<eos>"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def tokenize(text):
    return re.findall(r"[a-z]+|\d+", text.lower())


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN+LSTM image captioning model on COCO captions.")
    parser.add_argument("--coco-root", type=Path, default=Path("/export/data/dataset/COCO"))
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def find_annotation(coco_root, explicit):
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Annotation file not found: {explicit}")
        return explicit
    candidates = sorted(coco_root.rglob("captions_train*.json"))
    if not candidates:
        candidates = sorted(coco_root.rglob("captions_val*.json"))
    if not candidates:
        raise FileNotFoundError(f"No captions_train*.json or captions_val*.json found under {coco_root}")
    return candidates[0]


def build_image_index(coco_root):
    index = {}
    for dirpath, _, filenames in os.walk(coco_root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS and filename not in index:
                index[filename] = path
    return index


def load_caption_items(coco_root, annotation_path, train_limit, seed):
    data = json.loads(annotation_path.read_text())
    images = {item["id"]: item["file_name"] for item in data["images"]}
    image_index = build_image_index(coco_root)

    items = []
    for ann in data["annotations"]:
        filename = images.get(ann["image_id"])
        image_path = image_index.get(filename)
        if image_path is None:
            continue
        items.append({"image": image_path, "caption": ann["caption"]})

    if not items:
        raise RuntimeError("No image-caption pairs found. Check COCO image paths and annotation file.")

    rng = random.Random(seed)
    rng.shuffle(items)
    if train_limit > 0:
        items = items[:train_limit]
    return items


def build_vocab(items, vocab_size):
    counter = Counter()
    for item in items:
        counter.update(tokenize(item["caption"]))
    words = [word for word, _ in counter.most_common(max(0, vocab_size - 3))]
    itos = [PAD, UNK, EOS] + words
    stoi = {word: index for index, word in enumerate(itos)}
    return stoi, itos


class CaptionDataset(Dataset):
    def __init__(self, items, stoi, image_transform):
        self.items = items
        self.stoi = stoi
        self.transform = image_transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(item["image"]).convert("RGB")
        image = self.transform(image)
        tokens = tokenize(item["caption"]) + [EOS]
        caption = torch.tensor([self.stoi.get(token, self.stoi[UNK]) for token in tokens], dtype=torch.long)
        return image, caption, item["caption"], str(item["image"])


def collate_batch(batch):
    images, captions, references, paths = zip(*batch)
    max_len = max(len(caption) for caption in captions)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, caption in enumerate(captions):
        padded[i, : len(caption)] = caption
    return torch.stack(images), padded, list(references), list(paths)


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim, pretrained=True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        except AttributeError:
            resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(resnet.fc.in_features, embed_dim)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            features = self.features(images).flatten(1)
        return self.proj(features)


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_features, captions):
        embeddings = self.embedding(captions[:, :-1])
        inputs = torch.cat([image_features.unsqueeze(1), embeddings], dim=1)
        outputs, _ = self.lstm(inputs)
        return self.fc(outputs)

    def generate(self, image_features, itos, max_len=18):
        inputs = image_features.unsqueeze(1)
        states = None
        generated = []
        for _ in range(max_len):
            outputs, states = self.lstm(inputs, states)
            logits = self.fc(outputs[:, -1])
            next_id = int(logits.argmax(dim=1).item())
            word = itos[next_id] if next_id < len(itos) else UNK
            if word == EOS:
                break
            if word not in {PAD, UNK}:
                generated.append(word)
            inputs = self.embedding(torch.tensor([[next_id]], device=image_features.device))
        return " ".join(generated) if generated else "(empty)"


def train_one_epoch(encoder, decoder, loader, criterion, optimizer, device):
    encoder.train()
    decoder.train()
    running_loss = 0.0
    seen = 0
    for images, captions, _, _ in loader:
        images = images.to(device)
        captions = captions.to(device)
        features = encoder(images)
        logits = decoder(features, captions)
        loss = criterion(logits.reshape(-1, logits.size(-1)), captions.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        seen += batch_size
    return running_loss / seen


def save_training_curve(history):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross entropy loss")
    ax.set_title("Caption training loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.png", dpi=180)
    plt.close(fig)


def save_generated_examples(encoder, decoder, dataset, itos, device, sample_limit):
    encoder.eval()
    decoder.eval()
    rows = []
    fig, axes = plt.subplots(sample_limit, 1, figsize=(8, 2.5 * sample_limit))
    if sample_limit == 1:
        axes = [axes]

    with torch.no_grad():
        for i in range(sample_limit):
            image, _, reference, path = dataset[i]
            features = encoder(image.unsqueeze(0).to(device))
            generated = decoder.generate(features, itos)
            rows.append({"image": Path(path).name, "reference": reference, "generated": generated})

            raw = Image.open(path).convert("RGB")
            axes[i].imshow(raw)
            title = f"ref: {reference}\ngen: {generated}"
            axes[i].set_title("\n".join(textwrap.wrap(title, width=90)), fontsize=8)
            axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "generated_captions.png", dpi=180)
    plt.close(fig)

    with (DATA_DIR / "generated_captions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "reference", "generated"])
        writer.writeheader()
        writer.writerows(rows)


def write_vocab(itos):
    with (DATA_DIR / "vocab.txt").open("w", encoding="utf-8") as f:
        for index, word in enumerate(itos):
            f.write(f"{index}\t{word}\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    annotation = find_annotation(args.coco_root, args.annotation)
    items = load_caption_items(args.coco_root, annotation, args.train_limit, args.seed)
    stoi, itos = build_vocab(items, args.vocab_size)
    write_vocab(itos)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = CaptionDataset(items, stoi, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"annotation: {annotation}")
    print(f"pairs: {len(dataset)}")
    print(f"vocab size: {len(itos)}")

    encoder = CNNEncoder(args.embed_dim, pretrained=not args.no_pretrained).to(device)
    decoder = LSTMDecoder(len(itos), args.embed_dim, args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(
        list(encoder.proj.parameters()) + list(decoder.parameters()),
        lr=args.lr,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(encoder, decoder, loader, criterion, optimizer, device)
        history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f}")

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "itos": itos,
        },
        CKPT_DIR / "caption_cnn_lstm.pt",
    )
    with (DATA_DIR / "training_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history)

    save_training_curve(history)
    save_generated_examples(encoder, decoder, dataset, itos, device, min(args.sample_limit, len(dataset)))
    print(f"saved: {DATA_DIR / 'training_metrics.csv'}")
    print(f"saved: {DATA_DIR / 'generated_captions.csv'}")
    print(f"saved: {FIG_DIR / 'training_curve.png'}")
    print(f"saved: {FIG_DIR / 'generated_captions.png'}")


if __name__ == "__main__":
    main()
