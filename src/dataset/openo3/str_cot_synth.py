"""Generate synthetic structured chain-of-thought (CoT) traces for the Open-o3-Video STR dataset
using offline vLLM serving of a Qwen3-VL model.

Two stages per sample, batched across samples to keep the GPU saturated:
  Stage A (RCAP) — for every SAM3-tracked concept, overlay only that concept's bbox on the augmented frame stack and ask the model to caption the highlighted region.

  Stage B (CoT)  — overlay all concepts' bboxes, feed the per-concept RCAPs + question + GT answer into the CoT system prompt, and generate a structured reasoning trace ending in <answer>...</answer>.

The extracted answer is checked against the GT with a cheap heuristic (tag present, non-empty, token overlap) so obviously-broken generations are flagged; a stronger LLM-judge pass can run later over the stored traces. Results are written one JSON file per sample.

Mirrors Section 4 of scripts/vllm_serving/qwen3vl32b.ipynb. Usage:
    python src/dataset/openo3/str_cot_synth.py --config config/dataset/openo3_str_cot_synth.yaml
"""
import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import re
import json
import time
import logging
import argparse
import datetime
import tempfile
import socket

import cv2
import yaml
import numpy as np
import torch
from datasets import load_dataset
from qwen_vl_utils import fetch_video, process_vision_info
from torchcodec.encoders import VideoEncoder
from vllm import LLM, SamplingParams

from src.utils.openo3_utils import (
    get_video_path, splice_keyframes,
    load_track_manifest, load_track_masks, masks_to_bboxes,
)

logger = logging.getLogger("str_cot_synth")


# Sample the video at the track-run fps and splice keyframes in, reproducing the tracked frame stack.
def build_frame_stack(sample, fps):
    frames = fetch_video({"video": get_video_path(sample), "fps": fps})
    frames_list = list(frames.permute(0, 2, 3, 1).cpu().numpy().astype("uint8"))  # list of (H,W,3) RGB
    H, W = frames_list[0].shape[:2]
    splice_keyframes(sample, frames_list, H, W)
    return np.stack(frames_list), (H, W)


# Overlay the given concepts' per-frame tracked bboxes on a copy of the frame stack (all one color).
def annotate_concepts(frames_np, masks_dict, concepts, hw, color, thickness):
    annotated = frames_np.copy()
    for concept in concepts:
        bboxes = masks_to_bboxes(masks_dict[concept], target_size=hw)
        for t, (x1, y1, x2, y2) in enumerate(bboxes.astype(int)):
            if x2 > x1 and y2 > y1:
                cv2.rectangle(annotated[t], (x1, y1), (x2, y2), color=tuple(color), thickness=thickness)
    return annotated


# Uniformly subsample the stack to tgt_frames and write it to disk as an mp4 (Qwen3-VL's video
# processor loads from a file path, not a PIL frame list).
def encode_clip(frames_np, out_path, tgt_frames):
    T = len(frames_np)
    indices = np.linspace(0, T - 1, tgt_frames, dtype=int)
    clip = torch.tensor(frames_np[indices]).permute(0, 3, 1, 2).to(torch.uint8)
    VideoEncoder(clip, frame_rate=1).to_file(out_path)


# Build the vLLM input packet for a Qwen3-VL message containing one mp4 video, using the new
# video processor path (process_vision_info with patch_size=16, returning video kwargs).
def build_video_inputs(messages, processor):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, do_resize=False)
    _, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    return {
        "prompt": prompt,
        "multi_modal_data": {"video": video_inputs},
        "mm_processor_kwargs": video_kwargs,
    }


# Build the CoT user prompt from the per-concept region captions, question, and GT answer.
def create_cot_user_prompt(question, concept_captions, answer):
    prompt = "Red bounding box annotated objects/entities in the video:\n"
    for concept, caption in concept_captions.items():
        prompt += f"- {concept}: {caption}\n"
    prompt += f"Question: {question}\n"
    # prompt += f"Ground Truth Answer: {answer}\n"
    return prompt


# Extract the <answer>...</answer> span from a CoT trace; None if the tag is absent.
def extract_answer(cot_text):
    m = re.search(r"<answer>(.*?)</answer>", cot_text, re.DOTALL)
    return m.group(1).strip() if m else None


# Cheap heuristic answer check: tag present, non-empty, and token overlap with the GT above a floor.
# Flags obviously-broken generations; a stronger LLM-judge pass can run later over the stored traces.
def check_answer(extracted, gt_answer, min_overlap):
    if not extracted:
        return dict(passed=False, reason="no_answer_tag", overlap=0.0)
    pred_tokens = set(re.findall(r"\w+", extracted.lower()))
    gt_tokens = set(re.findall(r"\w+", gt_answer.lower()))
    if not pred_tokens:
        return dict(passed=False, reason="empty_answer", overlap=0.0)
    overlap = len(pred_tokens & gt_tokens) / len(gt_tokens) if gt_tokens else 0.0
    passed = overlap >= min_overlap
    return dict(passed=passed, reason="ok" if passed else "low_overlap", overlap=round(overlap, 3))


# Build the per-concept RCAP request packets (one per concept) for a single sample. Each concept's
# bbox is overlaid alone, the clip is written to tmp_dir, and a Qwen3-VL video packet is built.
def build_rcap_requests(sample, masks_dict, hw, frames_np, rcap_prompt, processor, vcfg, tmp_dir):
    
    requests = []
    for concept in masks_dict:
        annotated = annotate_concepts(frames_np, masks_dict, [concept], hw, vcfg["bbox_color"], vcfg["bbox_thickness"])
        
        concept_slug = concept.replace(" ", "_").replace("/", "_")
        clip_path = os.path.join(tmp_dir, f"{sample['id']}_{concept_slug}.mp4")
        
        encode_clip(annotated, clip_path, vcfg["tgt_frames"])
        
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": rcap_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Question: " + sample["question"] + "\n\nRelevant concept: " + concept
                    },
                    {
                        "type": "video",
                        "video": clip_path,
                        "min_pixels": vcfg["min_pixels"],
                        "max_pixels": vcfg["max_pixels"]
                    },
                ],
            }
        ]
        requests.append((concept, clip_path, build_video_inputs(messages, processor)))
    
    return requests


# Build the CoT request packet for a single sample from its generated per-concept captions. All
# concepts' bboxes are overlaid on one clip written to tmp_dir.
def build_cot_request(sample, masks_dict, hw, frames_np, concept_captions, cot_system_prompt,
                      processor, vcfg, tmp_dir):
    
    concepts = list(masks_dict.keys())
    annotated = annotate_concepts(frames_np, masks_dict, concepts, hw, vcfg["bbox_color"], vcfg["bbox_thickness"])
    
    clip_path = os.path.join(tmp_dir, f"{sample['id']}_cot.mp4")
    encode_clip(annotated, clip_path, vcfg["tgt_frames"])

    cot_user_prompt = create_cot_user_prompt(sample["question"], concept_captions, sample["answer"])
    
    messages = [
        {
            "role": "system", 
            "content": [
                {
                    "type": "text", 
                    "text": cot_system_prompt
                }
            ]
        },
        {
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text":  cot_user_prompt
                },
                {
                    "type": "video", 
                    "video": clip_path,
                    "min_pixels": vcfg["min_pixels"], 
                    "max_pixels": vcfg["max_pixels"]
                },
            ]
        },
    ]
    return clip_path, build_video_inputs(messages, processor)


# Load the STR subset, keep only samples with a saved masks file, and apply the test/num_samples cap.
# Returns (str_dataset, run_meta_track, track_samples).
def load_data(cfg, test):
    track_dir = cfg["track_results_dir"]
    dataset = load_dataset("json", data_files=cfg["dataset"]["json_path"], split="train")
    str_dataset = [s for s in dataset if s["task"] == cfg["dataset"]["task"]]
    run_meta_track, track_samples = load_track_manifest(track_dir)
    logger.info(f"Loaded {len(str_dataset)} STR samples; track manifest has {len(track_samples)} entries")

    # only keep samples that have a track entry with a saved masks file (something to caption)
    str_dataset = [s for s in str_dataset if s["id"] in track_samples
                   and track_samples[s["id"]].get("masks_file") is not None]

    # --test caps to 3 samples; otherwise honor the configured num_samples
    if test:
        str_dataset = str_dataset[:3]
    elif cfg["dataset"]["num_samples"] is not None:
        str_dataset = str_dataset[:cfg["dataset"]["num_samples"]]
    logger.info(f"Processing {len(str_dataset)} samples with track masks")

    return str_dataset, run_meta_track, track_samples


# Create and return (out_dir, samples_dir) for the run; appends _test to the exp dir in test mode.
def setup_dirs(cfg, test):
    exp_name = cfg["experiment_name"] + ("_test" if test else "")
    out_dir = os.path.join(cfg["results_root"], exp_name)
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    return exp_name, out_dir, samples_dir


# Write one sample's result (RCAPs, CoT, extracted answer, heuristic check) as its own JSON file.
def write_record(samples_dir, sample, concept_captions, concepts, cot_text, extracted, answer_check):
    record = dict(
        id=sample["id"],
        question=sample["question"],
        gt_answer=sample["answer"],
        concepts=concepts,
        concept_captions=concept_captions,
        cot=cot_text,
        extracted_answer=extracted,
        answer_check=answer_check,
    )
    json.dump(record, open(os.path.join(samples_dir, f"{sample['id']}.json"), "w"), indent=2)


# Write the run-level metadata JSON for reproducibility.
def write_run_meta(out_dir, exp_name, cfg, model_path, track_dir, rcap_prompt, cot_system_prompt,
                   t_start, num_samples, num_done, num_skipped):
    run_meta = dict(
        experiment_name=exp_name,
        start_time=datetime.datetime.fromtimestamp(t_start).isoformat(),
        end_time=datetime.datetime.now().isoformat(),
        elapsed_seconds=round(time.time() - t_start, 1),
        hostname=socket.gethostname(),
        model_path=model_path,
        config=cfg,
        track_results_dir=track_dir,
        rcap_system_prompt=rcap_prompt.strip(),
        cot_system_prompt=cot_system_prompt.strip(),
        num_samples=num_samples,
        num_done=num_done,
        num_skipped=num_skipped,
        script_path=os.path.abspath(__file__),
    )
    json.dump(run_meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test", action="store_true", help="run only 3 samples and write to a _test exp dir")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from transformers import AutoProcessor

    model_path = cfg["model"]["path"]
    rcap_prompt = open(cfg["prompts"]["rcap_system_prompt_path"]).read()
    cot_system_prompt = open(cfg["prompts"]["cot_system_prompt_path"]).read()

    # load the STR subset and the prior SAM3 track manifest
    track_dir = cfg["track_results_dir"]
    str_dataset, run_meta_track, track_samples = load_data(cfg, args.test)

    exp_name, out_dir, samples_dir = setup_dirs(cfg, args.test)
    skipped_path = os.path.join(out_dir, "skipped.json")

    # carry the track-run fps into the video cfg so build_frame_stack matches the tracked stack
    vcfg = dict(cfg["video"], fps=run_meta_track["config"]["video"]["fps"])

    # load the model once and build the two stage-specific samplers
    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(model=model_path, **cfg["llm"])
    rcap_sampling = SamplingParams(**cfg["sampling"]["rcap"])
    cot_sampling = SamplingParams(**cfg["sampling"]["cot"])

    skipped = []
    def flush_skipped():
        json.dump(skipped, open(skipped_path, "w"), indent=2)

    t_start = time.time()
    total_done = 0
    batch_size = cfg["batch_size"]

    # process the dataset in sample-batches; within a batch, RCAPs for all concepts run as one
    # generate call (Stage A), then all CoTs run as one generate call (Stage B)
    for b in range(0, len(str_dataset), batch_size):
        batch = str_dataset[b:b + batch_size]

        # build the frame stack + masks once per sample, reused across both stages
        prepared = []  # (sample, masks_dict, hw, frames_np)
        for sample in batch:
            try:
                frames_np, hw = build_frame_stack(sample, vcfg["fps"])
                masks_dict = load_track_masks(track_samples[sample["id"]], track_dir)
                if not masks_dict:
                    raise ValueError("empty masks_dict")
                prepared.append((sample, masks_dict, hw, frames_np))
            except Exception as e:
                skipped.append(dict(id=sample["id"], reason=f"prep_{type(e).__name__}: {e}"))
                flush_skipped()
                logger.error(f"{sample['id']}: prep failed ({e}), skipping")

        if not prepared:
            continue

        # annotated clips are materialized to a per-batch temp dir and removed after each stage
        tmp_dir = tempfile.mkdtemp(prefix="str_cot_", dir=out_dir)

        # --- Stage A: caption every concept of every sample in one batched generate ---
        rcap_inputs, rcap_index, rcap_clips = [], [], []  # rcap_index[i] = (sample_pos, concept)
        for pos, (sample, masks_dict, hw, frames_np) in enumerate(prepared):
            for concept, clip_path, req in build_rcap_requests(sample, masks_dict, hw, frames_np,
                                                               rcap_prompt, processor, vcfg, tmp_dir):
                rcap_inputs.append(req)
                rcap_index.append((pos, concept))
                rcap_clips.append(clip_path)

        rcap_outputs = llm.generate(rcap_inputs, sampling_params=rcap_sampling)
        for p in rcap_clips:
            if os.path.exists(p):
                os.remove(p)

        # regroup the flat RCAP outputs back into per-sample concept->caption dicts
        per_sample_captions = [dict() for _ in prepared]
        for (pos, concept), o in zip(rcap_index, rcap_outputs):
            per_sample_captions[pos][concept] = o.outputs[0].text.strip()

        # --- Stage B: build and run one CoT generate per sample, batched ---
        cot_inputs, cot_clips = [], []
        for pos, (sample, masks_dict, hw, frames_np) in enumerate(prepared):
            clip_path, req = build_cot_request(sample, masks_dict, hw, frames_np,
                                               per_sample_captions[pos], cot_system_prompt,
                                               processor, vcfg, tmp_dir)
            cot_inputs.append(req)
            cot_clips.append(clip_path)
        cot_outputs = llm.generate(cot_inputs, sampling_params=cot_sampling)
        for p in cot_clips:
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmp_dir)

        # extract the answer, run the heuristic check, and write one JSON per sample
        for pos, (sample, masks_dict, hw, frames_np) in enumerate(prepared):
            cot_text = cot_outputs[pos].outputs[0].text
            extracted = extract_answer(cot_text)
            answer_check = check_answer(extracted, sample["answer"], cfg["answer_check"]["min_overlap"])
            write_record(samples_dir, sample, per_sample_captions[pos], list(masks_dict.keys()),
                         cot_text, extracted, answer_check)
            if not answer_check["passed"]:
                logger.warning(f"{sample['id']}: answer_check failed ({answer_check['reason']})")

        total_done += len(prepared)
        logger.info(f"[{total_done}/{len(str_dataset)}] processed (batch ending at sample {b + len(batch)})")

    write_run_meta(out_dir, exp_name, cfg, model_path, track_dir, rcap_prompt, cot_system_prompt,
                   t_start, len(str_dataset), total_done, len(skipped))
    logger.info(f"Done. {total_done} done, {len(skipped)} skipped. Results in {out_dir}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
