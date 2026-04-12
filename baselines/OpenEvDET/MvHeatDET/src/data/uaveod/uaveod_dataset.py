"""
UAV-EOD Dataset for OpenEvDET/MvHeatDET

Adapts UAV-EOD dataset (raw event streams + APS frames) to OpenEvDET format.
"""

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import datapoints
from PIL import Image
import os
import json
import numpy as np

from src.core import register

__all__ = ["UAVEODDetection"]


def VoxelGrid(events, num_bins, height, width):
    """
    Convert raw event stream to voxel grid representation.

    Args:
        events: (N, 4) array with [x, y, polarity, timestamp]
        num_bins: number of temporal bins
        height, width: output dimensions

    Returns:
        (H, W, num_bins) voxel grid
    """
    if len(events) == 0:
        return np.zeros((height, width, num_bins), dtype=np.float32)

    # Normalize timestamps to [0, num_bins-1]
    ts = events[:, 3]
    ts_min, ts_max = ts.min(), ts.max()
    if ts_max - ts_min == 0:
        ts_norm = np.zeros_like(ts)
    else:
        ts_norm = (num_bins - 1) * (ts - ts_min) / (ts_max - ts_min)

    # Get integer and fractional parts
    ts_floor = np.floor(ts_norm).astype(np.int32)
    ts_ceil = np.ceil(ts_norm).astype(np.int32)
    ts_frac = ts_norm - ts_floor

    # Clip to valid range
    ts_floor = np.clip(ts_floor, 0, num_bins - 1)
    ts_ceil = np.clip(ts_ceil, 0, num_bins - 1)

    # Initialize voxel grid
    voxel = np.zeros((height, width, num_bins), dtype=np.float32)

    # Get coordinates
    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    pol = events[:, 2]  # polarity: -1 or 1

    # Clip to image bounds
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, pol, ts_floor, ts_ceil, ts_frac = x[valid], y[valid], pol[valid], ts_floor[valid], ts_ceil[valid], ts_frac[valid]

    # Bilinear temporal interpolation
    for i in range(len(x)):
        # Contribute to floor bin
        voxel[y[i], x[i], ts_floor[i]] += pol[i] * (1.0 - ts_frac[i])
        # Contribute to ceil bin (if different)
        if ts_floor[i] != ts_ceil[i]:
            voxel[y[i], x[i], ts_ceil[i]] += pol[i] * ts_frac[i]

    # Normalize to [0, 255] for visualization
    voxel_min, voxel_max = voxel.min(), voxel.max()
    if voxel_max - voxel_min > 0:
        voxel = 255 * (voxel - voxel_min) / (voxel_max - voxel_min)

    return voxel.astype(np.uint8)


@register
class UAVEODDetection(Dataset):
    """
    UAV-EOD Dataset for event-based object detection.

    Converts raw event streams to voxel grids and loads APS frames.
    Compatible with OpenEvDET/MvHeatDET training pipeline.
    """
    __inject__ = ["transforms"]

    def __init__(self, img_folder, event_folder, ann_folder, transforms,
                 num_bins=3, use_aps=False):
        """
        Args:
            img_folder: path to APS frames root (contains scene/subdataset structure)
            event_folder: path to event data root
            ann_folder: path to annotations root
            transforms: data augmentation pipeline
            num_bins: number of temporal bins for voxel grid
            use_aps: if True, use APS frames; if False, use voxel grid
        """
        super(Dataset, self).__init__()
        self._transforms = transforms
        self.img_folder = img_folder
        self.event_folder = event_folder
        self.ann_folder = ann_folder
        self.num_bins = num_bins
        self.use_aps = use_aps

        # UAV-EOD categories
        self.category2name = {
            0: 'car',
            1: 'two-wheel',
            2: 'pedestrian',
            3: 'bus',
            4: 'truck',
        }
        self.name2category = {v: k for k, v in self.category2name.items()}

        # Collect all samples from nested structure: scene/subdataset/frame
        self.samples = []
        for scene in sorted(os.listdir(img_folder)):
            scene_img_path = os.path.join(img_folder, scene)
            if not os.path.isdir(scene_img_path):
                continue

            for subdataset in sorted(os.listdir(scene_img_path)):
                subdataset_img_path = os.path.join(scene_img_path, subdataset)
                subdataset_event_path = os.path.join(event_folder, scene, subdataset)
                subdataset_ann_path = os.path.join(ann_folder, scene, subdataset)

                if not os.path.isdir(subdataset_img_path):
                    continue

                # Get all frames in this subdataset
                frames = sorted([f for f in os.listdir(subdataset_img_path) if f.endswith('.png')])
                for frame in frames:
                    self.samples.append({
                        'img_path': os.path.join(subdataset_img_path, frame),
                        'event_path': os.path.join(subdataset_event_path, frame.replace('.png', '.npy')),
                        'ann_path': os.path.join(subdataset_ann_path, frame.replace('.png', '.json')),
                    })

        print(f"[UAVEODDetection] Loaded {len(self.samples)} samples from {img_folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load APS frame
        aps_img = Image.open(sample['img_path']).convert("RGB")
        W, H = aps_img.size

        # Load and convert events to voxel grid
        try:
            events = np.load(sample['event_path'])  # (N, 4): [x, y, polarity, timestamp]
        except FileNotFoundError:
            print(f"[UAVEODDetection] Missing event file: {sample['event_path']}, using empty events")
            events = np.zeros((0, 4), dtype=np.float32)
        voxel = VoxelGrid(events, self.num_bins, H, W)

        # Convert voxel grid to RGB image (always ensure 3 channels)
        if voxel.shape[2] >= 3:
            voxel_rgb = voxel[:, :, :3]
        else:
            voxel_rgb = np.repeat(voxel[:, :, :1], 3, axis=2)
        voxel_img = Image.fromarray(voxel_rgb.astype(np.uint8), mode='RGB')

        # Choose input: APS or voxel grid
        img = aps_img if self.use_aps else voxel_img

        # Load annotations
        ann_path = sample['ann_path']
        try:
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"[UAVEODDetection] Missing/invalid annotation: {ann_path}, using empty")
            ann_data = {'shapes': []}

        # Parse annotations (UAV-EOD format: 'shapes' with 'points' and 'label')
        boxes = []
        labels = []
        for shape in ann_data.get('shapes', []):
            points = shape['points']
            label_name = shape.get('label', shape.get('lable', ''))  # Handle typo

            if label_name not in self.name2category:
                continue

            # Convert points to [x1, y1, x2, y2]
            x1, y1 = points[0]
            x2, y2 = points[2] if len(points) > 2 else points[1]
            boxes.append([x1, y1, x2, y2])
            labels.append(self.name2category[label_name])

        # Handle empty annotations
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd = torch.zeros((len(boxes_tensor),), dtype=torch.int64)

        # Wrap boxes as datapoints.BoundingBox so torchvision v2 transforms
        # (SanitizeBoundingBox, ConvertBox, etc.) can process them correctly
        boxes_dp = datapoints.BoundingBox(
            boxes_tensor,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=(H, W),
        )

        # Prepare target dict
        target = {
            'image_id': torch.tensor([idx]),
            'boxes': boxes_dp,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
            'orig_size': torch.tensor([H, W], dtype=torch.int64),
        }

        # Apply transforms
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


# Category mappings for compatibility
uaveod_category2name = {
    0: 'car',
    1: 'two-wheel',
    2: 'pedestrian',
    3: 'bus',
    4: 'truck',
}

uaveod_name2category = {v: k for k, v in uaveod_category2name.items()}
uaveod_category2label = {k: i for i, k in enumerate(uaveod_category2name.keys())}
uaveod_label2category = {v: k for k, v in uaveod_category2label.items()}
