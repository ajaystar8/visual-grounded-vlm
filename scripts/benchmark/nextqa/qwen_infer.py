# For now, use lmmseval (do not use this for now)
import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import numpy as np
import json
import socket
import datetime
import yaml
import argparse

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to inference config YAML")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

with open(args.config) as f:
    cfg = yaml.safe_load(f)

experiment_name = cfg["experiment_name"]
model_path = cfg["model"]["path"]

nextqa_root = cfg["dataset"]["root"]
nextqa_split = cfg["dataset"]["split"]
nextqa_system_prompt_path = cfg["dataset"]["system_prompt_path"]
with open(nextqa_system_prompt_path) as f:
    nextqa_system_prompt = f.read()

BATCH_SIZE = cfg["batch_size"]  # we need to do this to prevent RAM overflow

RESULTS_ROOT = os.path.join(cfg["results_root"], experiment_name)

LLM_CONFIG = cfg["llm"]

SAMPLING_CONFIG = cfg["sampling"]

VIDEO_CONFIG = cfg["video"]


def build_prompts(dataset, processor):
    """
    Generator that yields (llm_input, sample) one at a time.

    Args:
        dataset: HuggingFace dataset of NExT-QA annotations.
        processor: AutoProcessor for the model (applies chat template).

    Yields:
        tuple: (llm_inputs dict with 'prompt' and 'multi_modal_data', sample dict)
    """
    for sample in dataset:
        messages = [
             {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": nextqa_system_prompt
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sample["question"] + "\nOptions:\n" + sample["options_str"]
                    },
                    {
                        "type": "video",
                        "video": sample["video_path"],
                        **VIDEO_CONFIG,
                    },
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        _, video_inputs = process_vision_info(messages)
        if video_inputs is not None:
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


def save_metadata(run_dir, run_start_time, num_samples):
    metadata = {
        "run_start_time": run_start_time.isoformat(),
        "model_path": model_path,
        "experiment_name": experiment_name,
        "dataset": {
            "name": cfg["dataset"]["name"],
            "root": nextqa_root,
            "num_samples_processed": num_samples,
        },
        "system_prompt_path": nextqa_system_prompt_path,
        "system_prompt_template": nextqa_system_prompt.strip(),
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
    from src.dataset.nextqa import NextQADataset

    run_start_time = datetime.datetime.now()
    run_dir = make_run_dir(run_start_time)

    dataset = NextQADataset(split=nextqa_split, load_video=False)
    processor = AutoProcessor.from_pretrained(model_path)

    llm = LLM(model=model_path, **LLM_CONFIG)
    sampling_params = SamplingParams(**SAMPLING_CONFIG)

    results_path = os.path.join(run_dir, "results.json")
    results = []
    total_processed = 0

    prompt_gen = build_prompts(dataset, processor)
    while True:
        batch_inputs, batch_samples = [], []
        for llm_inputs, sample in prompt_gen:
            batch_inputs.append(llm_inputs)
            batch_samples.append(sample)
            if len(batch_inputs) == BATCH_SIZE:
                break

        if not batch_inputs:
            break

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        for sample, o in zip(batch_samples, outputs):
            results.append({
                "question": sample["question"],
                "gt_answer": sample["answer"],
                "gt_answer_option": sample["answer_option"],
                "vid": sample["vid"],
                "qid": sample["qid"],
                "type": sample["type"],
                "options_str": sample["options_str"],
                "video_path": sample["video_path"],
                "num_frames_used": sample["num_frames_used"],
                "generated_answer": o.outputs[0].text,
            })

        total_processed += len(batch_samples)
        print(f"Processed {total_processed} samples so far...")

        # write results incrementally so partial progress is not lost on failure
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    save_metadata(run_dir, run_start_time, total_processed)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main()
