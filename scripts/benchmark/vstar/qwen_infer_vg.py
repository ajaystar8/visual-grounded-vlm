"""
V-STaR visual grounding inference using Qwen2.5-VL vision-language model.

Like qwen_infer.py, but overlays precomputed visual grounding annotations
(masks, bboxes, outlines, or points) onto video frames before inference.

Expects:
  - A YAML config file (passed via --config) with:
    - model.path: HuggingFace model ID (e.g., Qwen/Qwen2.5-VL-7B-Instruct)
    - model.experiment_dir (optional): Path to an SFT experiment directory containing
      adapter_config.json + adapter_model.safetensors. If set, the LoRA adapter is
      loaded on top of the base model (base model and rank are read from
      adapter_config.json; model.path is ignored).
    - dataset.root: Path to V-STaR dataset root
    - dataset.annotation_file: Relative path to annotation JSON within dataset.root
    - dataset.system_prompt_path: Path to .md file containing the question template
    - visual_grounding.anns_root: Directory containing masks/<qid>.json annotation files
    - visual_grounding.ann_type: One of 'mask' | 'bbox' | 'outline' | 'point'
    - results_root: Base output directory for results
    - batch_size: Inference batch size
    - llm/sampling/video: Configuration dicts for LLM, sampling params, and video processing

Output:
  - Writes results.json with generated answers and metadata.json with config/run info

Controlled by:
  - bash_scripts/benchmark/vstar/qwen_infer_vg.sh
"""

import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import json
import socket
import datetime
import yaml
import argparse

import torch
import numpy as np

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from qwen_vl_utils import process_vision_info

from src.utils.annotation_utils import load_mask_annotations, annotate_frames

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to inference config YAML")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

with open(args.config) as f:
    cfg = yaml.safe_load(f)

experiment_name = cfg["experiment_name"]

lora_experiment_dir = cfg["model"].get("experiment_dir")
if lora_experiment_dir:
    with open(os.path.join(lora_experiment_dir, "adapter_config.json")) as f:
        adapter_cfg = json.load(f)
    model_path = adapter_cfg["base_model_name_or_path"]
    lora_max_rank = adapter_cfg["r"]
else:
    model_path = cfg["model"]["path"]
    lora_max_rank = None

vstar_root = cfg["dataset"]["root"]
vstar_annotation_path = os.path.join(vstar_root, cfg["dataset"]["annotation_file"])
vstar_system_prompt_path = cfg["dataset"]["system_prompt_path"]
with open(vstar_system_prompt_path) as f:
    vstar_system_prompt = f.read()

BATCH_SIZE = cfg["batch_size"]  # we need to do this to prevent RAM overflow

RESULTS_ROOT = os.path.join(cfg["results_root"], experiment_name)

VG_ANNOTATION_CONFIG = {
    "root": cfg["visual_grounding"]["anns_root"],
    "type": cfg["visual_grounding"]["ann_type"],
}

LLM_CONFIG = {
    "gpu_memory_utilization": cfg["llm"]["gpu_memory_utilization"],
    "enforce_eager": cfg["llm"]["enforce_eager"],
    "limit_mm_per_prompt": cfg["llm"]["limit_mm_per_prompt"],
}

SAMPLING_CONFIG = {
    "max_tokens": cfg["sampling"]["max_tokens"],
    "temperature": cfg["sampling"]["temperature"],
}

VIDEO_CONFIG = {
    "total_pixels": cfg["video"]["total_pixels"],
    "min_pixels": cfg["video"]["min_pixels"],
    "fps": cfg["video"]["fps"],
}


def build_prompts(dataset, processor, missing_annotations):
    """
    Generator that yields (llm_input, sample) one at a time.

    Args:
        dataset: HuggingFace dataset of V-STaR annotations.
        processor: AutoProcessor for the model (applies chat template).
        missing_annotations: list to append dicts for samples whose annotation
            file is absent (mutated in place).

    Yields:
        tuple: (llm_inputs dict with 'prompt' and 'multi_modal_data', sample dict)
    """
    for sample in dataset:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vstar_system_prompt.format(question=sample['question'])},
                    {
                        "type": "video",
                        "video": os.path.join(vstar_root, "videos", sample['vid'] + ".mp4"),
                        **VIDEO_CONFIG,
                    },
                ]
            },
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        _, video_inputs = process_vision_info(messages)
        if video_inputs is not None:

            # apply relevant VG annotations to video frames
            annotation_path = os.path.join(VG_ANNOTATION_CONFIG['root'], "masks", f"{sample['qid']}.json")
            if not os.path.exists(annotation_path):
                missing_annotations.append({"qid": sample["qid"], "vid": sample["vid"], "annotation_path": annotation_path})
                print(f"[WARN] Missing annotation file, skipping sample qid={sample['qid']}: {annotation_path}")
                continue

            annotated_inputs = []
            for video_input in video_inputs:
                record = load_mask_annotations(annotation_path)
                annotated_input = annotate_frames(video_input, record["masks_np"], annotation_type=VG_ANNOTATION_CONFIG['type'])
                annotated_input = torch.tensor(np.array(annotated_input)).permute(0, 3, 1, 2).to(torch.float32)
                annotated_inputs.append(annotated_input)

            video_inputs = annotated_inputs
            mm_data = {'video': video_inputs}
            sample["num_frames_used"] = video_inputs[0].shape[0]

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        yield llm_inputs, sample


def make_run_dir(run_start_time):
    run_name = run_start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_metadata(run_dir, run_start_time, num_samples, missing_annotations):
    metadata = {
        "run_start_time": run_start_time.isoformat(),
        "model_path": model_path,
        "lora_experiment_dir": lora_experiment_dir,
        "lora_max_rank": lora_max_rank,
        "experiment_name": experiment_name,
        "dataset": {
            "name": cfg["dataset"]["name"],
            "root": vstar_root,
            "annotation_file": vstar_annotation_path,
            "num_samples_processed": num_samples,
        },
        "missing_annotations": {
            "count": len(missing_annotations),
            "samples": missing_annotations,
        },
        "system_prompt_path": vstar_system_prompt_path,
        "system_prompt_template": vstar_system_prompt.strip(),
        "llm_config": LLM_CONFIG,
        "sampling_config": SAMPLING_CONFIG,
        "video_config": VIDEO_CONFIG,
        "script_path": os.path.abspath(__file__),
        "hostname": socket.gethostname(),
        "results_file": os.path.join(run_dir, "results.json"),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor

    run_start_time = datetime.datetime.now()
    run_dir = make_run_dir(run_start_time)

    dataset = load_dataset("json", data_files=vstar_annotation_path, split="train")
    processor = AutoProcessor.from_pretrained(model_path)

    llm_kwargs = dict(LLM_CONFIG)
    lora_request = None
    if lora_experiment_dir:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_max_rank
        lora_request = LoRARequest("sft_adapter", 1, lora_experiment_dir)

    llm = LLM(model=model_path, **llm_kwargs)
    sampling_params = SamplingParams(**SAMPLING_CONFIG)

    results_path = os.path.join(run_dir, "results.json")
    results = []
    total_processed = 0
    missing_annotations = []

    prompt_gen = build_prompts(dataset, processor, missing_annotations)
    while True:
        batch_inputs, batch_samples = [], []
        for llm_inputs, sample in prompt_gen:
            batch_inputs.append(llm_inputs)
            batch_samples.append(sample)
            if len(batch_inputs) == BATCH_SIZE:
                break

        if not batch_inputs:
            break

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params, lora_request=lora_request)
        for sample, o in zip(batch_samples, outputs):
            results.append({
                "qid": sample["qid"],
                "vid": sample["vid"],
                "num_frames_used": sample["num_frames_used"],
                "domain": sample["domain"],
                "question": sample["question"],
                "gt_answer": sample.get("answer", sample.get("gt_answer")),
                "generated_answer": o.outputs[0].text,
            })

        total_processed += len(batch_samples)
        print(f"Processed {total_processed} samples so far...")

        # write results incrementally so partial progress is not lost on failure
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    save_metadata(run_dir, run_start_time, total_processed, missing_annotations)
    if missing_annotations:
        print(f"[WARN] {len(missing_annotations)} samples skipped due to missing annotation files.")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main()
