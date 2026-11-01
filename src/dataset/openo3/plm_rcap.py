"""Generate Perception-LM-8B captions for every SAM3-tracked concept in the Open-o3-Video STR
dataset: a global video summary on the clean reconstructed video, plus a per-concept caption on a
video annotated with that concept's tracked bbox. Results are written one JSON file per sample.

All generations for a single sample (summary + per-concept captions) are batched into one
model.generate call. Annotated videos are written to temp files and deleted after each sample.

Usage:
    python src/dataset/openo3/plm_rcap.py --config config/dataset/openo3_plm_rcap.yaml
"""
import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import json
import time
import logging
import argparse
import datetime
import tempfile

import cv2
import yaml
import torch
import numpy as np
from datasets import load_dataset
from qwen_vl_utils import fetch_video
from torchcodec.encoders import VideoEncoder
from transformers import AutoProcessor, AutoModelForImageTextToText

from src.utils.openo3_utils import (
    get_video_path, splice_keyframes,
    load_track_manifest, load_track_masks, masks_to_bboxes,
)

logger = logging.getLogger("plm_rcap")


# Sample the video at the track-run fps and splice keyframes in, reproducing the tracked frame stack.
def build_frame_stack(sample, fps):
    frames = fetch_video({"video": get_video_path(sample), "fps": fps})
    frames_list = list(frames.permute(0, 2, 3, 1).cpu().numpy().astype("uint8"))  # list of (H,W,3) RGB
    H, W = frames_list[0].shape[:2]
    splice_keyframes(sample, frames_list, H, W)
    return np.stack(frames_list), (H, W)


# Uniformly subsample the stack to tgt_frames and write it to disk as an mp4 PLM can load.
def encode_video(frames_np, out_path, tgt_frames, encode_fps):
    T = len(frames_np)
    indices = np.linspace(0, T - 1, tgt_frames, dtype=int)
    clip = torch.tensor(frames_np[indices]).permute(0, 3, 1, 2).to(torch.uint8)
    VideoEncoder(clip, frame_rate=encode_fps).to_file(out_path)


# Overlay one concept's per-frame tracked bbox on a copy of the frame stack.
def annotate_concept(frames_np, masks, hw, color, thickness):
    annotated = frames_np.copy()
    bboxes = masks_to_bboxes(masks, target_size=hw)
    for t, (x1, y1, x2, y2) in enumerate(bboxes.astype(int)):
        if x2 > x1 and y2 > y1:
            cv2.rectangle(annotated[t], (x1, y1), (x2, y2), color=tuple(color), thickness=thickness)
    return annotated


# Capitalize the first letter and ensure a single trailing period.
def clean_response(text):
    t = text.strip()
    return (t[0].upper() + t[1:] if t else t).rstrip(".") + "."


# Run PLM generation over a batch of (video_path, prompt) pairs; returns one cleaned string per pair.
def plm_generate_batch(model, processor, items, tgt_frames, gen_cfg):
    conversations = [[{
        "role": "user",
        "content": [
            {"type": "video", "url": video_path},
            {"type": "text", "text": prompt},
        ],
    }] for video_path, prompt in items]
    inputs = processor.apply_chat_template(
        conversations, num_frames=tgt_frames, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt", padding=True,
        video_load_backend="decord",
    ).to(model.device).to(model.dtype)
    generate_ids = model.generate(
        **inputs, max_new_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"])
    responses = processor.batch_decode(
        generate_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return [clean_response(r) for r in responses]


# Build all temp videos for a sample, run the batched generation, and return the result record.
def process_sample(sample, sample_entry, track_dir, model, processor, vcfg, gen_cfg, prompts, tmp_dir):
    fps = vcfg["fps"]
    frames_np, hw = build_frame_stack(sample, fps)
    masks_dict = load_track_masks(sample_entry, track_dir)

    tmp_paths = []
    items = []  # (video_path, prompt), index 0 is the clean-video summary

    # clean video for the global summary
    clean_path = os.path.join(tmp_dir, f"{sample['id']}_clean.mp4")
    encode_video(frames_np, clean_path, vcfg["tgt_frames"], vcfg["encode_fps"])
    tmp_paths.append(clean_path)
    items.append((clean_path, prompts["caption"]))

    # one annotated video + rcap prompt per tracked concept
    concepts = list(masks_dict.keys())
    for concept in concepts:
        annotated = annotate_concept(frames_np, masks_dict[concept], hw,
                                     vcfg["bbox_color"], vcfg["bbox_thickness"])
        ann_path = os.path.join(tmp_dir, f"{sample['id']}_{concept}.mp4")
        encode_video(annotated, ann_path, vcfg["tgt_frames"], vcfg["encode_fps"])
        tmp_paths.append(ann_path)
        items.append((ann_path, prompts["rcap"]))

    try:
        outputs = plm_generate_batch(model, processor, items, vcfg["tgt_frames"], gen_cfg)
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)

    return dict(
        id=sample["id"],
        question=sample["question"],
        answer=sample["answer"],
        video_summary=outputs[0],
        concept_captions=dict(zip(concepts, outputs[1:])),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test", action="store_true", help="run only 3 samples and write to a _test exp dir")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device = {device}")

    # load the PLM model and processor
    model_path = cfg["plm"]["model_path"]
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    processor.tokenizer.padding_side = "left"  # left-pad for correct batched decoder-only generation
    model = AutoModelForImageTextToText.from_pretrained(model_path).to(device)
    logger.info(f"Loaded PLM model: {model_path}")

    # load the STR subset and the prior SAM3 track manifest
    track_dir = cfg["track_results_dir"]
    dataset = load_dataset("json", data_files=cfg["dataset"]["json_path"], split="train")
    str_dataset = [s for s in dataset if s["task"] == cfg["dataset"]["task"]]
    run_meta_track, track_samples = load_track_manifest(track_dir)
    logger.info(f"Loaded {len(str_dataset)} STR samples; track manifest has {len(track_samples)} entries")

    # only process samples that have a track entry (i.e. were covered by the track run)
    str_dataset = [s for s in str_dataset if s["id"] in track_samples]

    # --test caps to 3 samples; otherwise honor the configured num_samples
    if args.test:
        str_dataset = str_dataset[:3]
    elif cfg["dataset"]["num_samples"] is not None:
        str_dataset = str_dataset[:cfg["dataset"]["num_samples"]]
    logger.info(f"Processing {len(str_dataset)} samples with track entries")

    # append _test to the exp dir name in test mode to keep runs separate
    exp_name = cfg["experiment_name"] + ("_test" if args.test else "")
    out_dir = os.path.join(cfg["results_root"], exp_name)
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    skipped_path = os.path.join(out_dir, "skipped.json")

    # carry the track-run fps into the video cfg so build_frame_stack matches the tracked stack
    vcfg = dict(cfg["video"], fps=run_meta_track["config"]["video"]["fps"])
    gen_cfg = cfg["plm"]
    prompts = cfg["prompts"]

    # run-level metadata for reproducibility
    run_meta = dict(
        experiment_name=exp_name,
        start_time=datetime.datetime.now().isoformat(),
        config=cfg,
        device=device,
        track_results_dir=track_dir,
        num_samples=len(str_dataset),
    )

    # accumulate skipped samples (missing masks file or processing error); flushed as found
    skipped = []
    def flush_skipped():
        json.dump(skipped, open(skipped_path, "w"), indent=2)

    t_start = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, sample in enumerate(str_dataset):
            sample_id = sample["id"]
            sample_entry = track_samples[sample_id]

            # skip samples whose track run produced no masks file (nothing to caption)
            if sample_entry.get("masks_file") is None:
                skipped.append(dict(id=sample_id, reason="no_masks_file"))
                flush_skipped()
                logger.warning(f"[{i + 1}/{len(str_dataset)}] {sample_id}: no masks file, skipping")
                continue

            try:
                record = process_sample(sample, sample_entry, track_dir, model, processor,
                                        vcfg, gen_cfg, prompts, tmp_dir)
            except Exception as e:
                # isolate per-sample failures so the long run keeps going
                skipped.append(dict(id=sample_id, reason=f"{type(e).__name__}: {e}"))
                flush_skipped()
                logger.error(f"[{i + 1}/{len(str_dataset)}] {sample_id}: failed ({e}), skipping")
                continue

            # write each sample's output as it completes so progress survives a long run
            json.dump(record, open(os.path.join(samples_dir, f"{sample_id}.json"), "w"), indent=2)
            logger.info(f"[{i + 1}/{len(str_dataset)}] {sample_id}: "
                        f"summary + {len(record['concept_captions'])} concept caption(s)")

    run_meta["end_time"] = datetime.datetime.now().isoformat()
    run_meta["elapsed_seconds"] = round(time.time() - t_start, 1)
    run_meta["num_skipped"] = len(skipped)
    json.dump(run_meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=2)
    logger.info(f"Done. {len(skipped)} skipped. Results in {out_dir}")


if __name__ == "__main__":
    main()
