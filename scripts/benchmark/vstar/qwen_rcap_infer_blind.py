import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import json
import socket
import datetime
import yaml
import argparse

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image as to_pil

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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

with open(cfg["dataset"]["system_prompt_path"]) as f:
    vstar_system_prompt = f.read()

RESULTS_ROOT = os.path.join(cfg["results_root"], experiment_name)
RCAP_CAPTIONS_ROOT = cfg["rcap_captions_root"]

def build_prompts(dataset, processor):
    """
    Generator that yields (llm_input, sample) one at a time.

    Args:
        dataset: HuggingFace dataset of V-STaR annotations.
        processor: AutoProcessor for the model (applies chat template).

    Yields:
        tuple: (llm_inputs dict with 'prompt' and 'multi_modal_data', sample dict)
    """
    for sample in dataset:

        # load the region caption
        rcap_file_path = os.path.join(RCAP_CAPTIONS_ROOT, "samples", f"{sample['qid']}.json")
        with open(rcap_file_path) as f:
            rcap = json.load(f)['rcap']

        # create message packet
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": vstar_system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"Question: {sample["question"]}\nRegion Caption: {rcap}"
                    },
                ],
            },
        ]

        # process message packet
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # prepare model input packet
        llm_inputs = {
            "prompt": prompt,
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
        "lora_experiment_dir": lora_experiment_dir,
        "lora_max_rank": lora_max_rank,
        "experiment_name": experiment_name,
        "dataset": {
            **cfg["dataset"],
            "num_samples_processed": num_samples,
        },
        "system_prompt_template": vstar_system_prompt.strip(),
        "llm_config": cfg["llm"],
        "sampling_config": cfg["sampling"],
        "video_config": None,
        "script_path": os.path.abspath(__file__),
        "hostname": socket.gethostname(),
        "results_file": os.path.join(run_dir, "results.json"),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    from src.dataset.vstar import VStarDataset
    from transformers import AutoProcessor

    run_start_time = datetime.datetime.now()
    run_dir = make_run_dir(run_start_time)

    dataset = VStarDataset(**cfg['dataset'])
    processor = AutoProcessor.from_pretrained(model_path)

    llm_kwargs = dict(cfg["llm"])
    lora_request = None
    if lora_experiment_dir:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_max_rank
        lora_request = LoRARequest("sft_adapter", 1, lora_experiment_dir)

    llm = LLM(model=model_path, **llm_kwargs)
    sampling_params = SamplingParams(**cfg["sampling"])

    results_path = os.path.join(run_dir, "results.json")
    results = []
    total_processed = 0

    prompt_gen = build_prompts(dataset, processor)
    while True:
        batch_inputs, batch_samples = [], []
        for llm_inputs, sample in prompt_gen:
            batch_inputs.append(llm_inputs)
            batch_samples.append(sample)
            if len(batch_inputs) == cfg["batch_size"]:
                break

        if not batch_inputs:
            break

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params, lora_request=lora_request)
        for sample, o in zip(batch_samples, outputs):
            results.append({
                "qid": sample["qid"],
                "vid": sample["vid"],
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

    save_metadata(run_dir, run_start_time, total_processed)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main()
