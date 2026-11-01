"""Verify and refine synthetic STR-CoT traces with an offline-served Qwen3-4B LLM judge.

Loads the per-sample CoT JSONs produced by str_cot_synth.py and buckets them by their
heuristic `answer_check.reason`:
  - `ok`          -> already correct, kept as-is.
  - `low_overlap` -> ambiguous; an offline Qwen3-4B judge gives a binary same/not-same
                     score by comparing the GT answer with the extracted answer. Score 1 -> kept.
  - anything else -> dropped (e.g. no_answer_tag, empty_answer).

Kept samples are copied (with an added `binary_llm_judge_score` field) into a sibling
`refined_samples/` dir; dropped samples are recorded in `dropped.json`.

Usage:
    python src/dataset/openo3/cot_verify_refine.py --config config/dataset/cot_verify_refine.yaml
"""
import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import glob
import json
import time
import logging
import argparse
import datetime
import socket

import yaml
from vllm import LLM, SamplingParams

logger = logging.getLogger("cot_verify_refine")

# System prompt for the binary LLM judge (mirrors scripts/vllm_serving/qwen3_4b_judge.ipynb).
JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator judging whether a generated answer matches the ground-truth "
    "answer to a question about a video.\n"
    "Judge ONLY on correctness of the content, not on length, verbosity, or phrasing: a short "
    "answer that is correct should score the same as a long one that conveys the same information.\n"
    "Output exactly 1 if the generated answer conveys the same correct answer as the ground "
    "truth, and exactly 0 otherwise. Output only the single digit, with no explanation."
)


# Load every sample JSON under <samples_root>/samples, returning (sample_dict, path) pairs.
def load_samples(samples_root):
    samples = []
    for path in sorted(glob.glob(os.path.join(samples_root, "samples", "*.json"))):
        with open(path) as f:
            samples.append((json.load(f), path))
    logger.info(f"Loaded {len(samples)} samples from {samples_root}")
    return samples


# Build the judge user prompt comparing the GT answer with the extracted answer.
def build_judge_prompt(sample):
    return (
        f"Question: {sample['question']}\n\n"
        f"Ground-truth answer: {sample['gt_answer']}\n\n"
        f"Generated answer: {sample['extracted_answer']}\n\n"
        "Score (1 if same, 0 if not):"
    )


# Judge all low_overlap samples in one offline generate call. Returns a list of scores (1/0/None)
# aligned with the input samples.
def judge_samples(samples, llm, sampling, processor):
    prompts = []
    for s, _ in samples:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_prompt(s)},
        ]
        prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, sampling_params=sampling)
    scores = []
    for o in outputs:
        out = o.outputs[0].text.strip()
        scores.append(1 if out.startswith("1") else 0 if out.startswith("0") else None)
    return scores


# Write a kept sample's JSON into refined_dir, adding the binary judge score (None for `ok` samples).
def write_refined(refined_dir, sample, score):
    record = dict(sample, binary_llm_judge_score=score)
    json.dump(record, open(os.path.join(refined_dir, f"{sample['id']}.json"), "w"), indent=2)


# Write the run-level metadata JSON for reproducibility.
def write_run_meta(out_dir, cfg, t_start, counts):
    run_meta = dict(
        start_time=datetime.datetime.fromtimestamp(t_start).isoformat(),
        end_time=datetime.datetime.now().isoformat(),
        elapsed_seconds=round(time.time() - t_start, 1),
        hostname=socket.gethostname(),
        config=cfg,
        judge_system_prompt=JUDGE_SYSTEM_PROMPT,
        counts=counts,
        script_path=os.path.abspath(__file__),
    )
    json.dump(run_meta, open(os.path.join(out_dir, "refine_run_meta.json"), "w"), indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from transformers import AutoProcessor

    samples_root = cfg["samples_root"]
    refined_dir = os.path.join(samples_root, "refined_samples")
    os.makedirs(refined_dir, exist_ok=True)

    # bucket samples by their heuristic answer_check reason
    samples = load_samples(samples_root)
    correct = [(s, p) for s, p in samples if s["answer_check"]["reason"] == "ok"]
    low_overlap = [(s, p) for s, p in samples if s["answer_check"]["reason"] == "low_overlap"]
    other = [(s, p) for s, p in samples if s["answer_check"]["reason"] not in ("ok", "low_overlap")]
    logger.info(f"ok: {len(correct)}, low_overlap: {len(low_overlap)}, other(dropped): {len(other)}")

    # judge the low_overlap group with the offline-served LLM
    model_path = cfg["model"]["path"]
    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(model=model_path, **cfg["llm"])
    sampling = SamplingParams(**cfg["sampling"])
    scores = judge_samples(low_overlap, llm, sampling, processor)

    t_start = time.time()
    dropped = []

    # `ok` samples are kept unconditionally (no judge score)
    for s, _ in correct:
        write_refined(refined_dir, s, None)

    # low_overlap samples are kept only if judged 1; others (0 / unparsed None) are dropped
    n_rescued = 0
    for (s, p), score in zip(low_overlap, scores):
        if score == 1:
            write_refined(refined_dir, s, score)
            n_rescued += 1
        else:
            dropped.append(dict(id=s["id"], path=p, reason=f"judged_{score}"))

    # everything not `ok`/`low_overlap` is dropped with its heuristic reason
    for s, p in other:
        dropped.append(dict(id=s["id"], path=p, reason=s["answer_check"]["reason"]))

    json.dump(dropped, open(os.path.join(samples_root, "dropped.json"), "w"), indent=2)

    counts = dict(
        total=len(samples),
        kept_ok=len(correct),
        low_overlap=len(low_overlap),
        low_overlap_rescued=n_rescued,
        dropped=len(dropped),
    )
    write_run_meta(samples_root, cfg, t_start, counts)
    logger.info(f"Done. kept {len(correct) + n_rescued}, dropped {len(dropped)}. Refined dir: {refined_dir}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
