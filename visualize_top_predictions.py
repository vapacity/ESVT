import argparse
import bisect
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from dataset import build_dataset
from dataset.UAV_EOD.collate_fn import BatchImageCollateFuncion
from models import build_model, build_postprocessor
from models.ESVT.box_ops import box_cxcywh_to_xyxy, box_iou
from models.ESVT.utils import check_empty_target
from util.misc import dist_utils
from util.optim.ema import build_ema


def get_args_parser():
    parser = argparse.ArgumentParser("Visualize top prediction results", add_help=False)
    parser.add_argument('--dataset', default='UAV-EOD', type=str)
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--model', default='ESVT', type=str, choices=['ESVT'])
    parser.add_argument('--model_type', default='event', type=str)
    parser.add_argument('--event_rep', default='voxel', type=str, choices=['voxel'])
    parser.add_argument('--backbone', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--backbone_pretrained', default=False)
    parser.add_argument('--transformer_scale', default='hybrid_transformer_L', type=str,
                        choices=['hybrid_transformer_L', 'hybrid_transformer_X', 'hybrid_transformer_H'])
    parser.add_argument('--streaming_type', default='none', type=str)
    parser.add_argument('--num_top_queries', default=300, type=int)
    parser.add_argument('--use_focal_loss', default=True)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_val', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', required=True, type=str)
    parser.add_argument('--output_dir', default='outputs/top_predictions', type=str)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--score_thresh', default=0.3, type=float)
    return parser


def resolve_original_path(concat_dataset, global_idx: int) -> Path:
    dataset_idx = bisect.bisect_right(concat_dataset.cumulative_sizes, global_idx)
    sample_idx = global_idx if dataset_idx == 0 else global_idx - concat_dataset.cumulative_sizes[dataset_idx - 1]
    sub_dataset = concat_dataset.datasets[dataset_idx]
    local_idx = sample_idx % 100
    file_name = sub_dataset.aps_files[local_idx]
    return Path(sub_dataset.aps_dir) / file_name


def draw_boxes(image_bgr: np.ndarray, boxes: torch.Tensor, color, labels=None, scores=None):
    canvas = image_bgr.copy()
    if boxes.numel() == 0:
        return canvas

    boxes = boxes.detach().cpu().numpy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text_parts = []
        if labels is not None:
            text_parts.append(str(int(labels[idx])))
        if scores is not None:
            text_parts.append(f"{float(scores[idx]):.2f}")
        if text_parts:
            cv2.putText(canvas, " ".join(text_parts), (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def evaluate_image_quality(prediction, target, score_thresh: float):
    pred_scores = prediction['scores']
    keep = pred_scores >= score_thresh
    if keep.sum().item() == 0 or target['boxes'].numel() == 0:
        return None

    pred_boxes = prediction['boxes'][keep]
    pred_labels = prediction['labels'][keep]
    pred_scores = pred_scores[keep]
    gt_boxes = target['boxes']
    gt_labels = target['labels']

    ious, _ = box_iou(pred_boxes, gt_boxes)
    best_ious, gt_indices = ious.max(dim=1)
    label_match = pred_labels == gt_labels[gt_indices]
    matched_ious = best_ious[label_match]

    if matched_ious.numel() == 0:
        return None

    return {
        'mean_iou': matched_ious.mean().item(),
        'max_iou': matched_ious.max().item(),
        'num_matches': int(matched_ious.numel()),
        'num_preds': int(pred_boxes.shape[0]),
        'pred_boxes': pred_boxes,
        'pred_labels': pred_labels,
        'pred_scores': pred_scores,
    }


@torch.no_grad()
def main(args):
    dist_utils.setup_distributed(args.print_rank if hasattr(args, 'print_rank') else 0,
                                 args.print_method if hasattr(args, 'print_method') else 'builtin',
                                 seed=42)
    device = torch.device(args.device)

    dataset_val = build_dataset(mode='val', args=args)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_val,
        sampler=SequentialSampler(dataset_val),
        drop_last=False,
        num_workers=0,
        collate_fn=BatchImageCollateFuncion(),
    )

    model = build_model(args).to(device)
    ema = build_ema(model)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if 'ema' in checkpoint:
        ema.load_state_dict(checkpoint['ema'])
        model_for_eval = ema.module.to(device)
    else:
        model_for_eval = model
    model_for_eval.eval()

    postprocessor = build_postprocessor(args).to(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = []
    status = None

    for (images, events, targets), indexes in data_loader_val:
        images = images.to(device)
        events = events.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_img_ids = indexes[1]
        target_keep = [bool(t) for t in targets]

        if indexes[-1][-1] % 100 == 0 or status is None:
            pre_status = None
        else:
            if status and all(s is not None for s in status):
                pre_status = [(state[0].detach(), state[1].detach()) for state in status]
            else:
                pre_status = None

        outputs, filtered_targets, status = model_for_eval(events, targets=targets, pre_status=pre_status)
        if not check_empty_target(filtered_targets):
            continue

        kept_global_img_ids = [img_id for img_id, keep in zip(global_img_ids, target_keep) if keep]
        orig_target_sizes = torch.stack([t['orig_size'] for t in filtered_targets], dim=0)
        predictions = postprocessor(outputs, orig_target_sizes)

        for local_idx, (prediction, target, global_idx) in enumerate(zip(predictions, filtered_targets, kept_global_img_ids)):
            gt_boxes = box_cxcywh_to_xyxy(target['boxes'])
            gt_boxes = gt_boxes * target['orig_size'].repeat(2)
            target_for_eval = {
                'boxes': gt_boxes,
                'labels': target['labels'],
            }
            quality = evaluate_image_quality(prediction, target_for_eval, args.score_thresh)
            if quality is None:
                continue

            ranked.append({
                'global_idx': int(global_idx),
                'quality': quality,
                'gt_boxes': gt_boxes.detach().cpu(),
                'gt_labels': target['labels'].detach().cpu(),
            })

    ranked.sort(key=lambda item: (item['quality']['mean_iou'], item['quality']['max_iou'], item['quality']['num_matches']),
                reverse=True)
    top_items = ranked[:args.top_k]

    for rank, item in enumerate(top_items, start=1):
        original_path = resolve_original_path(dataset_val, item['global_idx'])
        image_bgr = cv2.imread(str(original_path))
        if image_bgr is None:
            print(f"Warning: failed to read {original_path}, skipping")
            continue
        image_bgr = draw_boxes(image_bgr, item['gt_boxes'], color=(0, 255, 0), labels=item['gt_labels'])
        image_bgr = draw_boxes(
            image_bgr,
            item['quality']['pred_boxes'],
            color=(0, 0, 255),
            labels=item['quality']['pred_labels'],
            scores=item['quality']['pred_scores'],
        )

        stem = original_path.stem
        out_name = (
            f"{rank:02d}_global{item['global_idx']:04d}_miou{item['quality']['mean_iou']:.3f}"
            f"_{stem}.png"
        )
        cv2.imwrite(str(output_dir / out_name), image_bgr)

    summary_path = output_dir / 'summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        for rank, item in enumerate(top_items, start=1):
            original_path = resolve_original_path(dataset_val, item['global_idx'])
            f.write(
                f"{rank:02d} global_idx={item['global_idx']} "
                f"mean_iou={item['quality']['mean_iou']:.4f} "
                f"max_iou={item['quality']['max_iou']:.4f} "
                f"matches={item['quality']['num_matches']} "
                f"preds={item['quality']['num_preds']} "
                f"file={original_path}\n"
            )

    print(f"Saved {len(top_items)} visualizations to {output_dir}")
    print(f"Summary written to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize top prediction results', parents=[get_args_parser()])
    parser.add_argument('--print-method', default='builtin', type=str)
    parser.add_argument('--print-rank', default=0, type=int)
    args = parser.parse_args()
    main(args)
