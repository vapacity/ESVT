import argparse
import json
from pathlib import Path

COCO_BBOX_METRICS = {
    "AP@[0.50:0.95]": 0,
    "AP@0.50": 1,
    "AP@0.75": 2,
    "AP Small": 3,
    "AP Medium": 4,
    "AP Large": 5,
    "AR@1": 6,
    "AR@10": 7,
    "AR@100": 8,
    "AR Small": 9,
    "AR Medium": 10,
    "AR Large": 11,
}


def load_jsonl_log(log_path: Path):
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"No valid records found in {log_path}")
    return records


def collect_series(records, key):
    xs, ys = [], []
    for idx, record in enumerate(records):
        if key not in record:
            continue
        x = record.get("epoch", idx)
        value = record[key]
        if isinstance(value, (int, float)):
            xs.append(x)
            ys.append(value)
    return xs, ys


def collect_coco_metric(records, metric_index):
    xs, ys = [], []
    for idx, record in enumerate(records):
        metrics = record.get("test_coco_eval_bbox")
        if not isinstance(metrics, list) or len(metrics) <= metric_index:
            continue
        x = record.get("epoch", idx)
        value = metrics[metric_index]
        if isinstance(value, (int, float)):
            xs.append(x)
            ys.append(value)
    return xs, ys


def plot_training_curves(records, save_path: Path, show: bool = False):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib could not be imported. "
            "Please use a Python environment with compatible matplotlib and numpy versions."
        ) from exc

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training and Validation Curves", fontsize=16)

    # 1. Total training loss
    ax = axes[0, 0]
    for key, label in [
        ("train_loss", "Total Loss"),
        ("train_loss_vfl", "VFL Loss"),
        ("train_loss_bbox", "BBox Loss"),
        ("train_loss_giou", "GIoU Loss"),
    ]:
        xs, ys = collect_series(records, key)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3, label=label)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.lines:
        ax.legend()

    # 2. Learning rate
    ax = axes[0, 1]
    xs, ys = collect_series(records, "train_lr")
    if xs:
        ax.plot(xs, ys, color="tab:orange", marker="o", linewidth=1.8, markersize=3)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True, linestyle="--", alpha=0.35)

    # 3. Validation AP
    ax = axes[1, 0]
    for label in ["AP@[0.50:0.95]", "AP@0.50", "AP@0.75", "AP Small", "AP Medium"]:
        xs, ys = collect_coco_metric(records, COCO_BBOX_METRICS[label])
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3, label=label)
    ax.set_title("Validation AP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Precision")
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.lines:
        ax.legend()

    # 4. Validation AR
    ax = axes[1, 1]
    for label in ["AR@1", "AR@10", "AR@100", "AR Small", "AR Medium"]:
        xs, ys = collect_coco_metric(records, COCO_BBOX_METRICS[label])
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3, label=label)
    ax.set_title("Validation AR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Recall")
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.lines:
        ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_default_output_path(log_path: Path) -> Path:
    return log_path.with_name("training_curves.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot key training and validation curves from an ESVT log.txt file."
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to the JSONL log.txt file generated during training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the output figure. Default: training_curves.png next to the log file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving.",
    )
    args = parser.parse_args()

    log_path = args.log_path.expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    records = load_jsonl_log(log_path)
    output_path = args.output.expanduser().resolve() if args.output else build_default_output_path(log_path)

    plot_training_curves(records, output_path, show=args.show)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
