"""Ground unique concepts on their keyframes and bidirectionally track them across the
1fps video for every spatio-temporal sample in the Open-o3-Video STR dataset, storing the
SAM3 tracker outputs (masks + bboxes) and run/concept/sample metadata per sample.

Mirrors Step 3 of rough_notebooks/openo3_track_filter.ipynb. Usage:
    python scripts/track/openo3_str_sam3_track.py --config config/track/openo3_str_sam3_track.yaml
"""
import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import json
import time
import argparse
import datetime

import yaml
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from qwen_vl_utils import fetch_video
from transformers import Sam3Model, Sam3Processor, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from src.utils.openo3_utils import get_video_path, splice_keyframes


# Sample the video at the configured fps and splice keyframes in as concept grounding seeds.
def build_frames_and_entities(sample, video_cfg):
    frames = fetch_video({"video": get_video_path(sample), "fps": video_cfg["fps"]})
    frames_list = list(frames.permute(0, 2, 3, 1).cpu().numpy().astype("uint8"))  # list of (H,W,3) RGB
    H, W = frames_list[0].shape[:2]
    n_sampled = len(frames_list)
    entities = splice_keyframes(sample, frames_list, H, W)
    return np.stack(frames_list), entities, n_sampled, (H, W)


# Ground one concept on its seed frame, pick the best-IoU mask, then track it bidirectionally.
def track_entity(frames_np, concept, seed_idx, gt_bbox, models, threshold, device, dtype):
    det_model, det_processor, trk_model, trk_processor = models
    T, H, W = frames_np.shape[:3]

    # ground the concept on the seed frame via text prompt
    inputs = det_processor(images=Image.fromarray(frames_np[seed_idx]), text=concept,
                           return_tensors="pt").to(device, dtype=dtype)
    with torch.inference_mode():
        outputs = det_model(**inputs)
    cand = det_processor.post_process_instance_segmentation(
        outputs, threshold=threshold, target_sizes=[(H, W)])[0]["masks"].cpu().numpy().astype(bool)
    if len(cand) == 0:
        return None

    # pick the candidate mask with highest IoU vs the GT bbox as the tracking seed
    bbox_region = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = gt_bbox
    bbox_region[y1:y2, x1:x2] = True
    ious = [(m & bbox_region).sum() / max((m | bbox_region).sum(), 1) for m in cand]
    seed_mask = cand[int(np.argmax(ious))]

    # seed the tracker on the chosen mask and propagate forward then backward
    session = trk_processor.init_video_session(video=frames_np, inference_device=device, dtype=dtype)
    trk_processor.add_inputs_to_inference_session(
        inference_session=session, frame_idx=seed_idx, obj_ids=1,
        input_masks=torch.from_numpy(seed_mask)[None].to(device))

    track_masks = np.zeros((T, H, W), dtype=bool)
    with torch.inference_mode():
        for reverse in (False, True):
            for out in trk_model.propagate_in_video_iterator(session, start_frame_idx=seed_idx, reverse=reverse):
                m = trk_processor.post_process_masks([out.pred_masks], original_sizes=[(H, W)], binarize=True)[0]
                track_masks[out.frame_idx] = m[0, 0].cpu().numpy().astype(bool)
    return track_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test", action="store_true", help="run only 3 samples and write to a _test exp dir")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"Using device = {device}")

    # load the SAM3 text-grounding detector and the video tracker
    model_id = cfg["sam3"]["model_id"]
    
    # detection model and processor
    det_model = Sam3Model.from_pretrained(model_id).to(device, dtype=dtype).eval()
    det_processor = Sam3Processor.from_pretrained(model_id)

    # tracking model and processor
    trk_model = Sam3TrackerVideoModel.from_pretrained(model_id).to(device, dtype=dtype).eval()
    trk_processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)
    
    models = (det_model, det_processor, trk_model, trk_processor)

    # load the STR subset of the dataset
    dataset = load_dataset("json", data_files=cfg["dataset"]["json_path"], split="train")
    str_dataset = [s for s in dataset if s["task"] == cfg["dataset"]["task"]]

    # --test caps to 3 samples; otherwise honor the configured num_samples
    if args.test:
        str_dataset = str_dataset[:3]
    elif cfg["dataset"]["num_samples"] is not None:
        str_dataset = str_dataset[:cfg["dataset"]["num_samples"]]
    print(f"Loaded {len(str_dataset)} samples")

    # append _test to the exp dir name in test mode to keep runs separate
    exp_name = cfg["experiment_name"] + ("_test" if args.test else "")
    out_dir = os.path.join(cfg["results_root"], exp_name)
    os.makedirs(out_dir, exist_ok=True)
    fps = cfg["video"]["fps"]
    threshold = cfg["sam3"]["det_threshold"]

    # run-level metadata for reproducibility
    run_meta = dict(
        experiment_name=exp_name,
        start_time=datetime.datetime.now().isoformat(),
        config=cfg,
        device=device,
        dtype=str(dtype),
        num_samples=len(str_dataset),
    )

    manifest = []
    t_start = time.time()
    for i, sample in enumerate(str_dataset):
        frames_np, entities, n_sampled, (H, W) = build_frames_and_entities(sample, cfg["video"])
        T = len(frames_np)

        # one entry per concept: track it, then record masks/bboxes (or found=False if grounding failed)
        sample_masks = {}  # concept -> (T,H,W) bool, saved to the per-sample npz
        concept_records = []
        for e in entities:
            track_masks = track_entity(frames_np, e["concept"], e["seed_idx"], e["gt_bbox"],
                                       models, threshold, device, dtype)
            found = track_masks is not None
            if found:
                sample_masks[e["concept"]] = track_masks
            concept_records.append(dict(
                concept=e["concept"],
                found=found,
                seed_idx=e["seed_idx"],
                gt_bbox=[e["gt_bbox"][0] / W, e["gt_bbox"][1] / H, e["gt_bbox"][2] / W, e["gt_bbox"][3] / H],  # normalized xyxy
            ))

        # store all masks of a sample together (compressed); skip the npz if nothing was found
        masks_file = None
        if sample_masks:
            masks_file = f"{sample['id']}_masks.npz"
            np.savez_compressed(os.path.join(out_dir, masks_file), **sample_masks)

        manifest.append(dict(
            id=sample["id"],
            source=sample.get("source", ""),
            question=sample["question"],
            answer=sample["answer"],
            frame_size=[H, W],
            n_sampled_frames=n_sampled,
            n_augmented_frames=T,
            fps=fps,
            masks_file=masks_file,
            concepts=concept_records,
        ))
        print(f"[{i + 1}/{len(str_dataset)}] {sample['id']}: "
              f"{sum(c['found'] for c in concept_records)}/{len(concept_records)} concepts tracked")

    run_meta["end_time"] = datetime.datetime.now().isoformat()
    run_meta["elapsed_seconds"] = round(time.time() - t_start, 1)

    # write the run metadata + per-sample manifest
    json.dump({"run": run_meta, "samples": manifest},
              open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
