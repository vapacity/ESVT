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
    parser.add_argument('--streaming_type', default='none', type=str,
                        choices=['none', 'stc', 'lstm', 'lstm_true', 'lstm_se', 'lstm_cbam', 'lstm_eca',
                                 'lstm_spatial', 'lstm_se_after', 'lstm_cbam_both'])
    parser.add_argument('--num_top_queries', default=300, type=int)
    parser.add_argument('--use_focal_loss', default=True)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_val', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', required=True, type=str)
    parser.add_argument('--output_dir', default='outputs/top_predictions', type=str)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--score_thresh', default=0.3, type=float)
    parser.add_argument('--mode', default='top', choices=['top', 'worst', 'random'],
                        help='Select best, worst, or random qualified samples')
    parser.add_argument('--scene_filter', default='all', choices=['all', 'normal', 'low_light', 'motion_blur'],
                        help='Filter samples by scene name in the file path')
    parser.add_argument('--seed', default=42, type=int)
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
    num_gt = int(gt_boxes.shape[0])
    num_preds = int(pred_boxes.shape[0])

    ious, _ = box_iou(pred_boxes, gt_boxes)

    # Greedy one-to-one matching on IoU, constrained by label agreement.
    candidates = []
    for pred_idx in range(num_preds):
        for gt_idx in range(num_gt):
            if int(pred_labels[pred_idx]) != int(gt_labels[gt_idx]):
                continue
            candidates.append((float(ious[pred_idx, gt_idx]), pred_idx, gt_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_preds = set()
    used_gts = set()
    matched_records = []
    for iou_value, pred_idx, gt_idx in candidates:
        if iou_value < 0.5:
            break
        if pred_idx in used_preds or gt_idx in used_gts:
            continue
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        matched_records.append((iou_value, pred_idx, gt_idx))

    if not matched_records:
        return None

    matched_ious = torch.tensor([m[0] for m in matched_records], dtype=pred_scores.dtype)
    matched_pred_indices = torch.tensor([m[1] for m in matched_records], dtype=torch.long)
    matched_gt_indices = torch.tensor([m[2] for m in matched_records], dtype=torch.long)

    num_matches = int(len(matched_records))
    num_gt_covered = int(len(used_gts))
    false_positives = max(num_preds - num_matches, 0)
    false_negatives = max(num_gt - num_gt_covered, 0)
    precision = num_matches / max(num_preds, 1)
    recall = num_gt_covered / max(num_gt, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'quality_score': 0.6 * f1 + 0.4 * float(matched_ious.mean().item()),
        'f1': f1,
        'mean_iou': matched_ious.mean().item(),
        'max_iou': matched_ious.max().item(),
        'min_iou': matched_ious.min().item(),
        'num_matches': num_matches,
        'num_gt': num_gt,
        'num_gt_covered': num_gt_covered,
        'num_preds': num_preds,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'score_mean': pred_scores.mean().item(),
        'score_max': pred_scores.max().item(),
        'score_min': pred_scores.min().item(),
        'pred_boxes': pred_boxes,
        'pred_labels': pred_labels,
        'pred_scores': pred_scores,
        'matched_pred_boxes': pred_boxes[matched_pred_indices],
        'matched_pred_labels': pred_labels[matched_pred_indices],
        'matched_pred_scores': pred_scores[matched_pred_indices],
        'matched_gt_indices': matched_gt_indices,
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
                'scene': resolve_original_path(dataset_val, int(global_idx)).parts[-3],
                'sequence': resolve_original_path(dataset_val, int(global_idx)).parts[-2],
                'quality': quality,
                'gt_boxes': gt_boxes.detach().cpu(),
                'gt_labels': target['labels'].detach().cpu(),
            })

    if args.scene_filter != 'all':
        ranked = [
            item for item in ranked
            if f"/{args.scene_filter}/" in str(resolve_original_path(dataset_val, item['global_idx']))
        ]

    if args.mode == 'top':
        ranked.sort(
            key=lambda item: (
                item['quality']['quality_score'],
                item['quality']['f1'],
                item['quality']['mean_iou'],
                item['quality']['num_gt_covered'],
            ),
            reverse=True,
        )
        selected_items = ranked[:args.top_k]
    elif args.mode == 'worst':
        ranked.sort(
            key=lambda item: (
                item['quality']['quality_score'],
                item['quality']['f1'],
                item['quality']['mean_iou'],
                item['quality']['num_gt_covered'],
            )
        )
        selected_items = ranked[:args.top_k]
    else:
        rng = np.random.default_rng(args.seed)
        if len(ranked) <= args.top_k:
            selected_items = ranked
        else:
            indices = rng.choice(len(ranked), size=args.top_k, replace=False)
            selected_items = [ranked[idx] for idx in indices]

    for rank, item in enumerate(selected_items, start=1):
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
            f"{rank:02d}_{args.mode}_global{item['global_idx']:04d}_miou{item['quality']['mean_iou']:.3f}"
            f"_{stem}.png"
        )
        cv2.imwrite(str(output_dir / out_name), image_bgr)

    summary_path = output_dir / 'summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        f.write(f"mode={args.mode} scene_filter={args.scene_filter} top_k={args.top_k} score_thresh={args.score_thresh}\n")
        f.write(
            "columns: rank global_idx scene sequence gt_count pred_count matched_pred matched_gt "
            "fp fn precision recall f1 quality_score mean_iou min_iou max_iou score_mean score_min score_max file\n"
        )
        for rank, item in enumerate(selected_items, start=1):
            original_path = resolve_original_path(dataset_val, item['global_idx'])
            f.write(
                f"{rank:02d} "
                f"global_idx={item['global_idx']} "
                f"scene={item['scene']} "
                f"sequence={item['sequence']} "
                f"gt_count={item['quality']['num_gt']} "
                f"pred_count={item['quality']['num_preds']} "
                f"matched_pred={item['quality']['num_matches']} "
                f"matched_gt={item['quality']['num_gt_covered']} "
                f"fp={item['quality']['false_positives']} "
                f"fn={item['quality']['false_negatives']} "
                f"precision={item['quality']['precision']:.4f} "
                f"recall={item['quality']['recall']:.4f} "
                f"f1={item['quality']['f1']:.4f} "
                f"quality_score={item['quality']['quality_score']:.4f} "
                f"mean_iou={item['quality']['mean_iou']:.4f} "
                f"min_iou={item['quality']['min_iou']:.4f} "
                f"max_iou={item['quality']['max_iou']:.4f} "
                f"score_mean={item['quality']['score_mean']:.4f} "
                f"score_min={item['quality']['score_min']:.4f} "
                f"score_max={item['quality']['score_max']:.4f} "
                f"file={original_path}\n"
            )

    print(f"Candidates after filtering: {len(ranked)}")
    print(f"Saved {len(selected_items)} visualizations to {output_dir}")
    print(f"Summary written to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize top prediction results', parents=[get_args_parser()])
    parser.add_argument('--print-method', default='builtin', type=str)
    parser.add_argument('--print-rank', default=0, type=int)
    args = parser.parse_args()
    main(args)
