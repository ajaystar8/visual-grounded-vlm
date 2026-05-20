"""
Generate pseudo-GT segmentation annotations for V-STaR by propagating SAM3 masks
from the GT bbox frame across all Qwen-sampled frames of each video.

Process:
1. Sample frames using process_vision_info (mimics Qwen inference: 1 fps, cap 768 frames).
2. Find the sampled frame closest to the first GT bbox frame.
3. Seed SAM3 with the GT bbox on that frame.
4. Propagate masks forward + backward across all sampled frames.
5. Save masks as COCO RLE in a per-video JSON + a run-level metadata.json.
"""

import argparse
import datetime
import json
import logging
import os
import socket
import sys

import decord
import numpy as np
import torch
from accelerate import Accelerator
from pycocotools import mask as mask_utils
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import smart_nframes

import datasets
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor


VIDEO_CONFIG = {
    "type": "video",
    "total_pixels": 16056320,  # 20480 * 28 * 28
    "min_pixels": 12544,       # 16 * 28 * 28
    "fps": 1.0,
}

SAM3_MODEL_ID = "facebook/sam3"

ANNOTATIONS_PATH = "/projects/torresani-lab/ajay/datasets/V-STaR/V_STaR_test_with_qid.json"
VIDEO_ROOT = "/projects/torresani-lab/ajay/datasets/V-STaR/videos"


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("propagate_sam3_masks")


def get_grounding_annotation(sample: dict, vid_path: str, vid_load_config: dict):
    """
    Returns the sampled-frame index closest to the first GT bbox and that bbox.

    Args:
        sample: dataset sample with 'bboxes' list (each entry: frame_index, xmin, ymin, xmax, ymax).
        vid_path: path to the video file.
        vid_load_config: dict passed to qwen smart_nframes (must include 'fps', etc.).

    Returns:
        (closest_idx: int, bbox: list[int]) where closest_idx indexes the Qwen-sampled frames.
    """
    vr = decord.VideoReader(vid_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes(vid_load_config, total_frames=total_frames, video_fps=video_fps)
    sampled_indices = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    gt_ann = sample["bboxes"][0]
    gt_frame_idx = gt_ann["frame_index"]
    gt_bbox = [gt_ann["xmin"], gt_ann["ymin"], gt_ann["xmax"], gt_ann["ymax"]]

    closest_idx = int(np.argmin(np.abs(np.array(sampled_indices) - gt_frame_idx)))
    error = abs(sampled_indices[closest_idx] - gt_frame_idx)
    assert error <= (video_fps), (
        f"Closest sampled frame too far from GT frame: error={error} frames at {video_fps} fps"
    )
    return closest_idx, gt_bbox


def get_video_path(video_root: str, video_id: str):
    path = os.path.join(video_root, f"{video_id}.mp4")
    return path if os.path.exists(path) else None


def encode_mask(mask_tensor: torch.Tensor) -> dict:
    """Encode a binary mask tensor (H, W) as COCO RLE. Returns JSON-serializable dict."""
    arr = mask_tensor.detach().to(torch.uint8).squeeze().cpu().numpy()
    assert arr.ndim == 2, f"expected 2D mask after squeeze, got shape {arr.shape}"
    rle = mask_utils.encode(np.asfortranarray(arr))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def annotate_sample(sample, video_root, model, processor, device, logger):
    """Run SAM3 propagation for one sample. Returns per-frame RLE masks dict or None on skip."""
    vid_id = sample["vid"]
    vid_path = get_video_path(video_root, vid_id)
    if vid_path is None:
        logger.warning(f"[{vid_id}] video file not found, skipping")
        return None

    vid_load_config = {"video": vid_path, **VIDEO_CONFIG}
    messages = [{"content": [vid_load_config]}]
    _, video_inputs = process_vision_info(messages)
    video_inputs = video_inputs[0]
    logger.info(f"[{vid_id}] sampled video tensor: {tuple(video_inputs.shape)}")

    closest_idx, gt_bbox = get_grounding_annotation(sample, vid_path, vid_load_config)
    logger.info(f"[{vid_id}] seed frame={closest_idx}, gt_bbox={gt_bbox}")

    inference_session = processor.init_video_session(
        video=video_inputs,
        inference_device=device,
        dtype=torch.bfloat16,
    )

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=closest_idx,
        obj_ids=1,
        input_boxes=[[gt_bbox]],
    )

    outputs = model(inference_session=inference_session, frame_idx=closest_idx)
    _ = processor.post_process_masks(
        [outputs.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
        binarize=True,
    )[0]

    video_segments = {}
    for reverse in (False, True):
        for out in model.propagate_in_video_iterator(inference_session, reverse=reverse):
            masks = processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=True,
            )[0]
            # masks has shape (num_obj, H, W); we only have one object
            video_segments[int(out.frame_idx)] = encode_mask(masks[0])

    logger.info(f"[{vid_id}] tracked {len(video_segments)} frames")
    return {
        "vid": vid_id,
        "qid": sample.get("qid"),
        "seed_frame_idx": closest_idx,
        "seed_bbox_xyxy": gt_bbox,
        "video_height": int(inference_session.video_height),
        "video_width": int(inference_session.video_width),
        "num_frames": len(video_segments),
        "masks": video_segments,  # {frame_idx: RLE}
    }


def save_metadata(run_dir, run_start_time, experiment_name, num_processed, num_skipped, num_failed):
    metadata = {
        "run_start_time": run_start_time.isoformat(),
        "run_end_time": datetime.datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "hostname": socket.gethostname(),
        "script_path": os.path.abspath(__file__),
        "sam3_model_id": SAM3_MODEL_ID,
        "video_config": VIDEO_CONFIG,
        "annotations_path": ANNOTATIONS_PATH,
        "video_root": VIDEO_ROOT,
        "output_dir": run_dir,
        "mask_format": "coco_rle",
        "num_samples_processed": num_processed,
        "num_samples_skipped": num_skipped,
        "num_samples_failed": num_failed,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Base output dir; <output_dir>/<experiment_name>/<timestamp> is created inside")
    parser.add_argument("--experiment_name", required=True, help="Experiment name; becomes a subdir under output_dir")
    args = parser.parse_args()

    logger = setup_logger()

    run_start_time = datetime.datetime.now()
    run_dir = os.path.join(args.output_dir, args.experiment_name, run_start_time.strftime("%Y-%m-%d_%H-%M-%S"))
    masks_dir = os.path.join(run_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    logger.info(f"Run dir: {run_dir}")

    device = Accelerator().device
    logger.info(f"Using device: {device}")
    model = Sam3TrackerVideoModel.from_pretrained(SAM3_MODEL_ID).to(device, dtype=torch.bfloat16)
    processor = Sam3TrackerVideoProcessor.from_pretrained(SAM3_MODEL_ID)

    dataset = datasets.load_dataset("json", data_files=ANNOTATIONS_PATH, split="train")
    logger.info(f"Loaded {len(dataset)} samples from {ANNOTATIONS_PATH}")

    num_processed = num_skipped = num_failed = 0
    for i, sample in enumerate(dataset):
        vid_id = sample["vid"]
        out_path = os.path.join(masks_dir, f"{sample.get('qid', vid_id)}.json")
        if os.path.exists(out_path):
            logger.info(f"[{vid_id}] already annotated, skipping")
            num_skipped += 1
            continue

        try:
            result = annotate_sample(sample, VIDEO_ROOT, model, processor, device, logger)
            if result is None:
                num_skipped += 1
                continue
            with open(out_path, "w") as f:
                json.dump(result, f)
            num_processed += 1
            logger.info(f"[{i + 1}/{len(dataset)}] saved {out_path}")
        except Exception as e:
            num_failed += 1
            logger.exception(f"[{vid_id}] failed: {e}")

    save_metadata(run_dir, run_start_time, args.experiment_name, num_processed, num_skipped, num_failed)
    logger.info(
        f"Done. processed={num_processed} skipped={num_skipped} failed={num_failed}. "
        f"Outputs in {run_dir}"
    )


if __name__ == "__main__":
    main()
