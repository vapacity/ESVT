#!/usr/bin/env python3
"""
统计EMRS-BAIDU数据集的事件流信息
"""

import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def load_json_label(label_path):
    """加载JSON标注文件"""
    with open(label_path, "r") as f:
        data = json.load(f)
    return data


def analyze_npy_file(npy_path):
    """分析单个npy文件
    格式: (N, 4) 数组，每行 [x, y, p, timestamp]
    """
    try:
        event_data = np.load(npy_path)

        if event_data is None or len(event_data) == 0:
            return None

        event_data = np.asarray(event_data)
        if event_data.ndim != 2 or event_data.shape[1] != 4:
            print(f"Unexpected npy format {npy_path}: shape {event_data.shape}")
            return None

        xs = event_data[:, 0]
        ys = event_data[:, 1]
        ps = event_data[:, 2]
        ts = event_data[:, 3]

        positive = np.sum(ps == 1)
        negative = np.sum(ps == 0)

        stats = {
            "num_events": len(event_data),
            "timestamp_min": float(ts.min()),
            "timestamp_max": float(ts.max()),
            "x_min": int(xs.min()),
            "x_max": int(xs.max()),
            "y_min": int(ys.min()),
            "y_max": int(ys.max()),
            "positive_events": int(positive),
            "negative_events": int(negative),
            "polarization_ratio": positive / len(event_data)
            if len(event_data) > 0
            else 0,
            "timestamp_range": float(ts.max() - ts.min()),
            "spatial_coverage_x": int(xs.max() - xs.min() + 1),
            "spatial_coverage_y": int(ys.max() - ys.min() + 1),
        }

        return stats
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None

        stats = {
            "num_events": len(event_data),
            "timestamp_min": float(event_data["timestamp"].min())
            if "timestamp" in event_data.dtype.names
            else 0,
            "timestamp_max": float(event_data["timestamp"].max())
            if "timestamp" in event_data.dtype.names
            else 0,
            "x_min": int(event_data["x"].min()) if "x" in event_data.dtype.names else 0,
            "x_max": int(event_data["x"].max()) if "x" in event_data.dtype.names else 0,
            "y_min": int(event_data["y"].min()) if "y" in event_data.dtype.names else 0,
            "y_max": int(event_data["y"].max()) if "y" in event_data.dtype.names else 0,
        }

        if "p" in event_data.dtype.names:
            positive = np.sum(event_data["p"] == 1)
            negative = np.sum(event_data["p"] == 0)
            stats["positive_events"] = int(positive)
            stats["negative_events"] = int(negative)
            stats["polarization_ratio"] = (
                positive / len(event_data) if len(event_data) > 0 else 0
            )

        stats["timestamp_range"] = stats["timestamp_max"] - stats["timestamp_min"]
        stats["spatial_coverage_x"] = stats["x_max"] - stats["x_min"] + 1
        stats["spatial_coverage_y"] = stats["y_max"] - stats["y_min"] + 1

        return stats
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None


def analyze_label_file(label_path):
    """分析单个JSON标注文件"""
    try:
        data = load_json_label(label_path)
        objects = data.get("shapes", [])

        boxes = []
        labels = []
        for obj in objects:
            points = obj.get("points", [])
            if len(points) >= 2:
                x1, y1 = points[0]
                x2, y2 = points[2]
                boxes.append([x1, y1, x2, y2])
                labels.append(obj.get("label", obj.get("lable", "unknown")))

        if boxes:
            boxes = np.array(boxes)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]

            return {
                "num_objects": len(objects),
                "avg_box_area": float(np.mean(areas)),
                "avg_box_width": float(np.mean(widths)),
                "avg_box_height": float(np.mean(heights)),
                "labels": labels,
                "categories": list(set(labels)),
            }
        return {"num_objects": 0, "labels": [], "categories": []}
    except Exception as e:
        print(f"Error loading {label_path}: {e}")
        return {"num_objects": 0, "labels": [], "categories": []}


def analyze_dataset(dataset_path, mode="train"):
    """分析整个数据集"""
    images_root = Path(dataset_path) / mode / "images"
    events_root = Path(dataset_path) / mode / "events"
    labels_root = Path(dataset_path) / mode / "labels"

    if not images_root.exists():
        print(f"Dataset path not found: {images_root}")
        return None

    scenes = sorted(
        [d for d in os.listdir(images_root) if os.path.isdir(images_root / d)]
    )

    stats = {
        "total_scenes": len(scenes),
        "total_subdatasets": 0,
        "total_frames": 0,
        "total_events": 0,
        "total_objects": 0,
        "scene_stats": [],
        "label_stats": {
            "category_counts": defaultdict(int),
            "objects_per_frame": [],
        },
        "event_stats": {
            "events_per_frame": [],
            "timestamp_ranges": [],
            "polarization_ratios": [],
        },
    }

    print(f"\n{'=' * 60}")
    print(f"Analyzing {mode.upper()} dataset: {dataset_path}")
    print(f"{'=' * 60}")

    for scene in tqdm(scenes, desc=f"Scanning {mode} scenes"):
        scene_images = images_root / scene
        scene_events = events_root / scene
        scene_labels = labels_root / scene

        sub_datasets = sorted(
            [d for d in os.listdir(scene_images) if os.path.isdir(scene_images / d)]
        )

        scene_info = {
            "scene": scene,
            "num_subdatasets": len(sub_datasets),
            "total_frames": 0,
            "total_events": 0,
        }

        for sub_dataset in sub_datasets:
            stats["total_subdatasets"] += 1

            sub_images = scene_images / sub_dataset
            sub_events = scene_events / sub_dataset
            sub_labels = scene_labels / sub_dataset

            if not sub_events.exists():
                continue

            npy_files = sorted(
                [f for f in os.listdir(sub_events) if f.endswith(".npy")]
            )

            for npy_file in npy_files:
                frame_id = npy_file.replace(".npy", "")

                npy_path = sub_events / npy_file
                label_path = sub_labels / f"{frame_id}.json"
                img_path = sub_images / f"{frame_id}.png"

                event_stats = analyze_npy_file(npy_path)
                if event_stats:
                    stats["total_events"] += event_stats["num_events"]
                    stats["event_stats"]["events_per_frame"].append(
                        event_stats["num_events"]
                    )
                    stats["event_stats"]["timestamp_ranges"].append(
                        event_stats["timestamp_range"]
                    )
                    if "polarization_ratio" in event_stats:
                        stats["event_stats"]["polarization_ratios"].append(
                            event_stats["polarization_ratio"]
                        )

                if label_path.exists():
                    label_stats = analyze_label_file(label_path)
                    stats["total_objects"] += label_stats["num_objects"]
                    stats["label_stats"]["objects_per_frame"].append(
                        label_stats["num_objects"]
                    )
                    for label in label_stats["labels"]:
                        stats["label_stats"]["category_counts"][label] += 1

                stats["total_frames"] += 1
                scene_info["total_frames"] += 1
                if event_stats:
                    scene_info["total_events"] += event_stats["num_events"]

        scene_info["avg_events_per_frame"] = (
            scene_info["total_events"] / scene_info["total_frames"]
            if scene_info["total_frames"] > 0
            else 0
        )
        stats["scene_stats"].append(scene_info)

    return stats


def print_statistics(stats, mode="train"):
    """打印统计结果"""
    if stats is None:
        return

    print(f"\n{'=' * 60}")
    print(f" {mode.upper()} DATASET STATISTICS")
    print(f"{'=' * 60}")

    print(f"\n【基本信息】")
    print(f"  场景数量 (Scenes):          {stats['total_scenes']}")
    print(f"  子数据集数量 (Sub-datasets): {stats['total_subdatasets']}")
    print(f"  总帧数 (Total frames):       {stats['total_frames']}")
    print(f"  总事件数 (Total events):     {stats['total_events']:,}")
    print(f"  总目标数 (Total objects):    {stats['total_objects']:,}")

    print(f"\n【事件统计】")
    if stats["event_stats"]["events_per_frame"]:
        events = np.array(stats["event_stats"]["events_per_frame"])
        print(f"  每帧事件数 - 平均值: {np.mean(events):,.1f}")
        print(f"  每帧事件数 - 中位数: {np.median(events):,.1f}")
        print(f"  每帧事件数 - 标准差: {np.std(events):,.1f}")
        print(f"  每帧事件数 - 最小值: {np.min(events):,}")
        print(f"  每帧事件数 - 最大值: {np.max(events):,}")
        print(f"  每帧事件数 - 25%:   {np.percentile(events, 25):,.1f}")
        print(f"  每帧事件数 - 75%:   {np.percentile(events, 75):,.1f}")

    print(f"\n【目标统计】")
    if stats["label_stats"]["objects_per_frame"]:
        objects = np.array(stats["label_stats"]["objects_per_frame"])
        print(f"  每帧目标数 - 平均值: {np.mean(objects):.2f}")
        print(f"  每帧目标数 - 中位数: {np.median(objects):.1f}")
        print(f"  每帧目标数 - 最大值: {np.max(objects)}")

    print(f"\n【类别分布】")
    if stats["label_stats"]["category_counts"]:
        sorted_categories = sorted(
            stats["label_stats"]["category_counts"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for cat, count in sorted_categories:
            pct = (
                count / stats["total_objects"] * 100
                if stats["total_objects"] > 0
                else 0
            )
            print(f"  {cat:20s}: {count:6,} ({pct:5.1f}%)")

    if stats["event_stats"]["polarization_ratios"]:
        pol = np.array(stats["event_stats"]["polarization_ratios"])
        print(f"\n【极性分布】")
        print(f"  正事件平均比例: {np.mean(pol) * 100:.1f}%")
        print(f"  正事件范围: [{np.min(pol) * 100:.1f}%, {np.max(pol) * 100:.1f}%]")

    print(f"\n【场景详情】")
    print(
        f"  {'Scene':<25s} | {'SubDS':>5s} | {'Frames':>7s} | {'Events':>12s} | {'Avg/Frame':>10s}"
    )
    print(f"  {'-' * 70}")
    for scene_info in stats["scene_stats"]:
        print(
            f"  {scene_info['scene']:<25s} | {scene_info['num_subdatasets']:>5d} | "
            f"{scene_info['total_frames']:>7d} | {scene_info['total_events']:>12,} | "
            f"{scene_info['avg_events_per_frame']:>10,.0f}"
        )


def save_statistics(stats, output_path):
    """保存统计结果到JSON文件"""

    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, defaultdict):
            return dict(obj)
        return obj

    import copy

    stats_copy = copy.deepcopy(stats)

    for key in ["event_stats", "label_stats"]:
        if key in stats_copy:
            for subkey in stats_copy[key]:
                if isinstance(stats_copy[key][subkey], list):
                    stats_copy[key][subkey] = [
                        convert_numpy(x) for x in stats_copy[key][subkey]
                    ]
                elif isinstance(stats_copy[key][subkey], defaultdict):
                    stats_copy[key][subkey] = dict(stats_copy[key][subkey])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_copy, f, indent=2, ensure_ascii=False)

    print(f"\n统计结果已保存到: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EMRS-BAIDU数据集统计分析")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="path to your dataset",
        help="数据集根目录路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="分析模式",
    )
    parser.add_argument(
        "--output", type=str, default="dataset_statistics.json", help="输出JSON文件路径"
    )

    args = parser.parse_args()

    if args.dataset_path == "path to your dataset":
        print("请设置数据集路径: --dataset_path /path/to/EMRS-BAIDU")
        print("示例: python stat_dataset.py --dataset_path /data/EMRS-BAIDU --mode all")
        exit(1)

    all_stats = {}

    modes = ["train", "val", "test"] if args.mode == "all" else [args.mode]

    for mode in modes:
        stats = analyze_dataset(args.dataset_path, mode)
        if stats:
            all_stats[mode] = stats
            print_statistics(stats, mode)

    if len(modes) > 1:
        print(f"\n{'=' * 60}")
        print(" ALL MODES SUMMARY")
        print(f"{'=' * 60}")
        total_frames = sum(s["total_frames"] for s in all_stats.values())
        total_events = sum(s["total_events"] for s in all_stats.values())
        total_objects = sum(s["total_objects"] for s in all_stats.values())
        print(f"  Total Frames:   {total_frames:,}")
        print(f"  Total Events:   {total_events:,}")
        print(f"  Total Objects:  {total_objects:,}")

    save_statistics(all_stats, args.output)
