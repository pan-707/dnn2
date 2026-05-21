import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


ANALOGIES = [
    ("king", "man", "woman", "queen"),
    ("brother", "man", "woman", "sister"),
    ("paris", "france", "japan", "tokyo"),
    ("rome", "italy", "france", "paris"),
    ("berlin", "germany", "italy", "rome"),
    ("better", "good", "bad", "worse"),
    ("bigger", "big", "small", "smaller"),
]

NEAREST_WORDS = ["king", "japan", "computer", "food", "music"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run word embedding analogy experiments.")
    parser.add_argument(
        "--model-name",
        default="glove-wiki-gigaword-50",
        help="Small public pretrained embedding available from gensim-data.",
    )
    parser.add_argument("--topn", type=int, default=5)
    return parser.parse_args()


def load_vectors(model_name):
    os.environ.setdefault("GENSIM_DATA_DIR", str(MODEL_DIR))
    import gensim.downloader as api

    print(f"loading model: {model_name}")
    print(f"gensim data dir: {os.environ['GENSIM_DATA_DIR']}")
    return api.load(model_name)


def format_results(results):
    return " / ".join(f"{word}:{score:.4f}" for word, score in results)


def run_analogies(model, topn):
    rows = []
    for positive_word, negative_word, add_word, expected in ANALOGIES:
        words = [positive_word, negative_word, add_word]
        missing = [word for word in words if word not in model]
        if missing:
            rows.append(
                {
                    "formula": f"{positive_word} - {negative_word} + {add_word}",
                    "expected": expected,
                    "top1": "",
                    "top1_score": "",
                    "top_results": f"missing: {', '.join(missing)}",
                    "hit_top5": "False",
                }
            )
            continue

        results = model.most_similar(
            positive=[positive_word, add_word],
            negative=[negative_word],
            topn=topn,
        )
        top1, top1_score = results[0]
        rows.append(
            {
                "formula": f"{positive_word} - {negative_word} + {add_word}",
                "expected": expected,
                "top1": top1,
                "top1_score": f"{top1_score:.6f}",
                "top_results": format_results(results),
                "hit_top5": str(expected in [word for word, _ in results]),
            }
        )
    return rows


def run_nearest_words(model, topn):
    rows = []
    for query in NEAREST_WORDS:
        if query not in model:
            rows.append({"query": query, "nearest": "missing"})
            continue
        rows.append({"query": query, "nearest": format_results(model.most_similar(query, topn=topn))})
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = load_vectors(args.model_name)
    analogy_rows = run_analogies(model, args.topn)
    nearest_rows = run_nearest_words(model, args.topn)

    write_csv(
        DATA_DIR / "analogies.csv",
        analogy_rows,
        ["formula", "expected", "top1", "top1_score", "top_results", "hit_top5"],
    )
    write_csv(DATA_DIR / "nearest_words.csv", nearest_rows, ["query", "nearest"])

    print("Analogy results:")
    for row in analogy_rows:
        print(
            f"{row['formula']} -> {row['top1']} "
            f"(expected: {row['expected']}, hit_top5={row['hit_top5']})"
        )
    print(f"saved: {DATA_DIR / 'analogies.csv'}")
    print(f"saved: {DATA_DIR / 'nearest_words.csv'}")


if __name__ == "__main__":
    main()
