"""
Helpers for loading SAM3 pseudo-GT annotations and overlaying them on frames.

Mask files are produced by `src/dataset/vstar/propagate_sam3_masks.py` and stored as JSON with
COCO-RLE-encoded per-frame masks.
"""

from PIL import Image
import matplotlib
import json

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils


RED_RGB = (255, 0, 0)  # frames are RGB; OpenCV draws into them with RGB tuples


def load_mask_annotations(json_path: str) -> dict:
    """
    Load a per-sample annotation file written by propagate_sam3_masks.py.

    Returns:
        dict with the original metadata fields and an extra `masks_np` field:
        {frame_idx: np.ndarray[H, W] uint8} — decoded binary masks ready for
        `annotate_frames(..., annotation_type='mask')`.
    """
    with open(json_path) as f:
        record = json.load(f)

    masks_np = {}
    for frame_idx_str, rle in record["masks"].items():
        rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
        masks_np[int(frame_idx_str)] = mask_utils.decode(rle)
    record["masks_np"] = masks_np
    return record


def _mask_to_numpy(mask) -> np.ndarray:
    """Coerce a mask (torch tensor [1,1,H,W] / [H,W] or numpy array) to uint8 (H, W)."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().to(torch.uint8).squeeze().cpu().numpy()
    else:
        mask = np.asarray(mask).squeeze().astype(np.uint8)
    assert mask.ndim == 2, f"expected 2D mask after squeeze, got shape {mask.shape}"
    return mask


def _find_center(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    left, right = xs.min(), xs.max()
    upper, lower = ys.min(), ys.max()

    x_mid = int(np.median([left, right]))
    exist_y = ys[xs == x_mid]
    center1 = (x_mid, int(np.median(exist_y))) if len(exist_y) > 0 else None

    y_mid = int(np.median([upper, lower]))
    exist_x = xs[ys == y_mid]
    center2 = (int(np.median(exist_x)), y_mid) if len(exist_x) > 0 else None

    if center1 and center2:
        avg = (int((center1[0] + center2[0]) / 2), int((center1[1] + center2[1]) / 2))
        if mask[avg[1], avg[0]] > 0:
            return avg
    if center1 and mask[center1[1], center1[0]] > 0:
        return center1
    if center2 and mask[center2[1], center2[0]] > 0:
        return center2
    return ((left + right) // 2, (upper + lower) // 2)


def annotate_frames(frames, video_segments, annotation_type="mask", **kwargs):
    """
    Overlay annotations on video frames.

    Args:
        frames: iterable of (C, H, W) torch tensors OR (H, W, C) numpy arrays.
        video_segments: dict {frame_idx -> mask}, where mask is either a torch
            tensor (any of [H,W] / [1,H,W] / [1,1,H,W]) or a (H, W) numpy array.
            The dict returned by `load_mask_annotations(...)["masks_np"]` works
            directly.
        annotation_type: 'mask' | 'bbox' | 'outline' | 'point'.

    kwargs by type:
        mask:    alpha=0.4, blur_kernel=21
        bbox:    thickness=2
        outline: thickness=2
        point:   marker_size=60
    """
    annotated_frames = []
    for idx, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        frame_np = np.asarray(frame)
        if frame_np.ndim == 3 and frame_np.shape[0] == 3:
            frame_np = frame_np.transpose(1, 2, 0)
        frame_np = frame_np.astype(np.uint8)

        if idx not in video_segments:
            annotated_frames.append(frame_np)
            continue

        mask = _mask_to_numpy(video_segments[idx])
        if mask.shape != frame_np.shape[:2]:
            mask = cv2.resize(mask, (frame_np.shape[1], frame_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        if annotation_type == "mask":
            alpha = kwargs.get("alpha", 0.4)
            blur_kernel = kwargs.get("blur_kernel", 21)
            frame_f = frame_np.astype(np.float32)
            soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_kernel, blur_kernel), 0)
            soft_mask = (soft_mask * alpha)[..., np.newaxis]
            colored_mask = np.full_like(frame_f, RED_RGB)
            annotated_frame = (frame_f * (1 - soft_mask) + colored_mask * soft_mask).clip(0, 255).astype(np.uint8)

        elif annotation_type == "bbox":
            thickness = kwargs.get("thickness", 2)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                annotated_frames.append(frame_np)
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            annotated_frame = cv2.rectangle(frame_np.copy(), (int(x1), int(y1)), (int(x2), int(y2)), RED_RGB, thickness)

        elif annotation_type == "outline":
            thickness = kwargs.get("thickness", 2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            annotated_frame = cv2.drawContours(frame_np.copy(), contours, -1, RED_RGB, thickness)

        elif annotation_type == "point":
            marker_size = kwargs.get("marker_size", 60)
            center = _find_center(mask)
            if center is None:
                annotated_frames.append(frame_np)
                continue
            radius = marker_size // 2
            annotated_frame = frame_np.copy()
            cv2.circle(annotated_frame, center, radius, RED_RGB, -1)
            font_scale = marker_size / 60
            cv2.putText(
                annotated_frame, "+",
                (center[0] - int(9 * font_scale), center[1] + int(7 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), int(font_scale * 2), cv2.LINE_AA,
            )

        else:
            raise ValueError(f"Unknown annotation_type: {annotation_type}")

        annotated_frames.append(annotated_frame)

    return annotated_frames

def overlay_masks(image, results, mode="mask"):
    """
    Overlay masks, bboxes, or outlines for multiple objects onto a single image.

    Unlike `annotate_frames`, which annotates one object per frame across a video
    using a {frame_idx -> mask} dict, this function annotates N objects on a single
    image using a batch results dict with stacked tensors.

    Args:
        image: PIL Image or BGR numpy array (e.g. from cv2.imread).
        results: dict with:
            - "masks": (N, H, W) bool/uint8 tensor  [required for mode="mask"/"outline"]
            - "boxes": (N, 4) float tensor, pixel-coord xyxy  [required for mode="bbox"]
        mode: "mask" | "bbox" | "outline"
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.convert("RGBA")

    n = len(results["masks"] if mode == "mask" else results["boxes"])
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n)]

    if mode == "mask":
        masks = 255 * results["masks"].cpu().numpy().astype(np.uint8)
        for mask, color in zip(masks, colors):
            mask = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            overlay.putalpha(mask.point(lambda v: int(v * 0.5)))
            image = Image.alpha_composite(image, overlay)

    elif mode == "bbox":
        boxes = results["boxes"].cpu().numpy()
        img_np = np.array(image.convert("RGB"))
        for bbox, color in zip(boxes, colors):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        image = Image.fromarray(img_np).convert("RGBA")

    elif mode == "outline":
        masks = results["masks"].cpu().numpy().astype(np.uint8)
        img_np = np.array(image.convert("RGB"))
        for mask, color in zip(masks, colors):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_np, contours, -1, color, 2)
        image = Image.fromarray(img_np).convert("RGBA")

    return image
