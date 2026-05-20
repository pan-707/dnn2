import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run pretrained object detectors on COCO images.")
    parser.add_argument("--image-root", type=Path, default=Path("/export/data/dataset/COCO"))
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def collect_images(root, limit):
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")

    images = []
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
                if len(images) >= limit:
                    return images
    if not images:
        raise RuntimeError(f"No image files found under {root}")
    return images


def load_model(name, device):
    if name == "fasterrcnn":
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn(weights=weights)
    elif name == "ssdlite":
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        model = ssdlite320_mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unknown model: {name}")

    model.eval().to(device)
    return model, weights.transforms(), weights.meta["categories"]


def run_model(model, transform, images, device):
    tensors = [transform(image).to(device) for image in images]
    with torch.no_grad():
        outputs = model(tensors)
    return [{key: value.cpu() for key, value in output.items()} for output in outputs]


def draw_detections(image, output, categories, threshold, max_detections):
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    boxes = output["boxes"]
    labels = output["labels"]
    scores = output["scores"]

    kept = 0
    for box, label, score in zip(boxes, labels, scores):
        score_value = float(score.item())
        if score_value < threshold:
            continue
        if kept >= max_detections:
            break

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        class_id = int(label.item())
        class_name = categories[class_id] if class_id < len(categories) else str(class_id)
        caption = f"{class_name} {score_value:.2f}"
        color = (255, 70, 70)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_box = draw.textbbox((x1, y1), caption, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), caption, fill=(255, 255, 255), font=font)
        kept += 1
    return canvas, kept


def save_prediction_grid(image_paths, model_outputs, categories, model_name, threshold, max_detections):
    rows = len(image_paths)
    fig, axes = plt.subplots(rows, 2, figsize=(9, 3.3 * rows))
    if rows == 1:
        axes = axes.reshape(1, 2)

    for row, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        drawn, _ = draw_detections(
            image,
            model_outputs[row],
            categories,
            threshold,
            max_detections,
        )
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(path.name, fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(drawn)
        axes[row, 1].set_title(model_name, fontsize=9)
        axes[row, 1].axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{model_name}_predictions.png", dpi=180)
    plt.close(fig)


def write_summary(image_paths, outputs_by_model, categories_by_model, threshold):
    with (DATA_DIR / "detection_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "image", "detections_over_threshold", "top_detections"])
        for model_name, outputs in outputs_by_model.items():
            categories = categories_by_model[model_name]
            for path, output in zip(image_paths, outputs):
                names = []
                count = 0
                for label, score in zip(output["labels"], output["scores"]):
                    score_value = float(score.item())
                    if score_value < threshold:
                        continue
                    class_id = int(label.item())
                    class_name = categories[class_id] if class_id < len(categories) else str(class_id)
                    names.append(f"{class_name}:{score_value:.2f}")
                    count += 1
                    if len(names) >= 5:
                        break
                writer.writerow([model_name, path.name, count, " / ".join(names)])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"image_root: {args.image_root}")

    image_paths = collect_images(args.image_root, args.num_images)
    images = [Image.open(path).convert("RGB") for path in image_paths]
    print("images:")
    for path in image_paths:
        print(f"- {path}")

    outputs_by_model = {}
    categories_by_model = {}
    for model_name in ["fasterrcnn", "ssdlite"]:
        print(f"running {model_name}")
        model, transform, categories = load_model(model_name, device)
        outputs = run_model(model, transform, images, device)
        outputs_by_model[model_name] = outputs
        categories_by_model[model_name] = categories
        save_prediction_grid(
            image_paths,
            outputs,
            categories,
            model_name,
            args.score_threshold,
            args.max_detections,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(image_paths, outputs_by_model, categories_by_model, args.score_threshold)
    print(f"saved: {FIG_DIR / 'fasterrcnn_predictions.png'}")
    print(f"saved: {FIG_DIR / 'ssdlite_predictions.png'}")
    print(f"saved: {DATA_DIR / 'detection_summary.csv'}")


if __name__ == "__main__":
    main()
