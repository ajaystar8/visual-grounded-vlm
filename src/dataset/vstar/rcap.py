"""Generate question-conditioned region captions (RCAP) for the V-STaR test set with an
in-process vLLM Qwen3-VL-32B model. For each sample, the SAM3 pseudo-GT mask is overlaid on the
decoded frames as a red bbox (handled by VStarDataset), 16 frames are uniformly subsampled and
passed as a list of images, and the frames + question are sent to the model with the rcap system
prompt. The generated caption is written to one JSON file per sample.

Loads the model in-process (no separate server) and runs batched llm.generate calls, mirroring
scripts/benchmark/vstar/qwen_infer_vg.py. Usage:
    python src/dataset/vstar/rcap.py --config config/dataset/vstar_rcap.yaml
"""
import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import json
import time
import yaml
import socket
import logging
import argparse
import datetime

import torch
import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from qwen_vl_utils import process_vision_info
from torchcodec.encoders import VideoEncoder

from src.dataset.vstar import VStarDataset

logger = logging.getLogger("vstar_rcap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def build_prompts(dataset, processor, tgt_frames, tmp_dir, SYSTEM_PROMPT):

    for sample in dataset:

        # use the annotated frames
        # HACK: Qwen3 has patch_size to be 16, however, the dataset class uses a patch size of 14 (default for qwen2.5)
        # Since patch size only affects the min/max pixels, and we are not near the extremums, I will let this slide for now. 
        frames = sample["frames"]

        # uniformly subsample the already annotated frames 
        indices = np.linspace(0, frames.shape[0] - 1, tgt_frames, dtype=int)
        frames = frames[indices]

        # Encode the annotated frames to get a video path
        tmp_video = os.path.join(tmp_dir, f"{sample['qid']}.mp4")
        VideoEncoder(frames.to(torch.uint8), frame_rate=1).to_file(tmp_video)

        # construct message packet
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample['question']},
                    {
                        "type": "video",
                        "video": tmp_video,
                        "min_pixels": 128 * 28 * 28, 
                        "max_pixels": 512 * 28 * 28
                    },
                ],
            },
        ]

        # apply chat template
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, do_resize=False)

        # Qwen3 uses new video processor (load the sampled annotated video)
        _, video_inputs, video_kwargs = process_vision_info(messages, 
                                                            image_patch_size=16, 
                                                            return_video_kwargs=True, 
                                                            return_video_metadata=True)

        # for logging
        sample["num_frames_used"] = frames.shape[0]
        sample["tmp_video"] = tmp_video

        yield {
            "prompt": prompt,
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": video_kwargs,
        }, sample


def setup_dirs(cfg):
    """Create and return output directories for the run."""
    exp_name = cfg["experiment_name"]
    out_dir = os.path.join(cfg["results_root"], exp_name)
    samples_out_dir = os.path.join(out_dir, "samples")
    tmp_dir = os.path.join(out_dir, "_tmp_videos")
    os.makedirs(samples_out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    return out_dir, samples_out_dir, tmp_dir


def main():

    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config
    cfg = yaml.safe_load(open(args.config))
    out_dir, samples_out_dir, tmp_dir = setup_dirs(cfg)

    # experiment specific config
    model_path = cfg["model"]["path"]
    exp_name = cfg["experiment_name"]
    tgt_frames = cfg["tgt_frames"]
    batch_size = cfg["batch_size"]

    # load dataset and system prompt (message packet constructed again, hence system prompt is required)
    dataset = VStarDataset(**cfg["dataset"])
    logger.info(f"Loaded {len(dataset)} V-STaR samples")

    system_prompt_path = cfg["dataset"]["system_prompt_path"]
    with open(system_prompt_path, 'r') as f:
        SYSTEM_PROMPT = f.read()

    # load the model and processor
    llm = LLM(model=model_path, **cfg["llm"])
    sampling_params = SamplingParams(**cfg["sampling"])
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info(f"Loaded model: {model_path}")

    run_start_time = datetime.datetime.now()
    
    total_processed = 0
    t_start = time.time()

    # pull batches off the generator and run one llm.generate per batch
    prompt_gen = build_prompts(dataset, processor, tgt_frames, tmp_dir, SYSTEM_PROMPT)
    while True:
        batch_inputs, batch_samples = [], []
        for llm_inputs, sample in prompt_gen:
            batch_inputs.append(llm_inputs)
            batch_samples.append(sample)
            if len(batch_inputs) == batch_size:
                break

        if not batch_inputs:
            break

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        # write each sample's caption as its own JSON file
        for sample, o in zip(batch_samples, outputs):
            record = dict(
                qid=sample["qid"],
                vid=sample["vid"],
                num_frames_used=sample["num_frames_used"],
                question=sample["question"],
                answer=sample.get("answer"),
                rcap=o.outputs[0].text,
            )
            json.dump(record, open(os.path.join(samples_out_dir, f"{record['qid']}.json"), "w"), indent=2)
            
            # remove the temp mp4 once its caption is written
            os.remove(sample["tmp_video"])

        total_processed += len(batch_samples)
        logger.info(f"[{total_processed}/{len(dataset)}] batch done")

    # run-level metadata for reproducibility
    run_meta = dict(
        experiment_name=exp_name,
        run_start_time=run_start_time.isoformat(),
        end_time=datetime.datetime.now().isoformat(),
        elapsed_seconds=round(time.time() - t_start, 1),
        model_path=model_path,
        hostname=socket.gethostname(),
        script_path=os.path.abspath(__file__),
        config=cfg,
        num_samples_processed=total_processed,
    )
    json.dump(run_meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=2)
    logger.info(f"Done. Processed {total_processed} samples. Results in {out_dir}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()