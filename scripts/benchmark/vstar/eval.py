"""
V-STaR evaluation using Qwen2.5-72B-Instruct for semantic answer scoring.

Expects:
  - --exp_dir: Directory containing results.json with generated answers from inference
  - results.json format: List of dicts with keys: question, gt_answer, generated_answer, etc.

Output:
  - Evaluates semantic similarity between generated and ground-truth answers using Qwen
  - Scores range 0-3 (0=no similarity, 1=low, 2=high, 3=complete)
  - Writes evaluation scores back to results.json

Controlled by:
  - bash_scripts/benchmark/vstar/eval.sh
"""

import os
import json
import time
import logging
import pandas as pd
import os.path as osp

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
MAX_TOKENS = 512

system_prompt = """
As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, \
based solely on the semantic similarity between the groundtruth and the candidate answer, \
disregarding any grammatical differences.
A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
Your response should be a single integer from 0, 1, 2, or 3.
"""
tmpl = 'Question: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: '

def build_prompts(data):
    
    prompts = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for sample in data:
        question = sample["question"]
        gt_ans = sample["gt_answer"]
        gen_ans = sample["generated_answer"]
        user_prompt = tmpl.format(question, gt_ans, gen_ans)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)

    return prompts

def score_to_verdict(score):
    return None if score == -1 else score >= 2

def llm_as_judge(data):

    prompts = build_prompts(data)

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    scores = []
    for response in outputs:
        try:
            score = int(response.outputs[0].text)
        except (ValueError, TypeError):
            score = -1
        scores.append(score)

    return scores

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Official V-STaR Benchmark Evaluation Script")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to an inference run directory containing metadata.json and results.json.",
    )
    args = parser.parse_args()

    inference_metadata_path = osp.join(args.exp_dir, 'metadata.json')
    result_json_path = osp.join(args.exp_dir, 'results.json')
    with open(inference_metadata_path, 'r') as f:
        inference_metadata = json.load(f)
    exp_name = inference_metadata['experiment_name']

    experiment_dir = osp.join(args.exp_dir, 'benchmark')
    os.makedirs(experiment_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    logger = logging.getLogger(__name__)

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    start_perf = time.time()
    metadata = {
        'exp_name': exp_name,
        'exp_dir': osp.abspath(args.exp_dir),
        'result_json_path': osp.abspath(result_json_path),
        'experiment_dir': experiment_dir,
        'model_name': MODEL_NAME,
        'max_tokens': MAX_TOKENS,
        'tensor_parallel_size': torch.cuda.device_count(),
        'cuda_device_count': torch.cuda.device_count(),
        'system_prompt': system_prompt,
        'prompt_template': tmpl,
        'start_time': start_time,
        'inference_metadata': inference_metadata,
    }
    logger.info("Experiment dir: %s", experiment_dir)
    logger.info("Metadata:\n%s", json.dumps(metadata, indent=2))

    with open(result_json_path, 'r') as f:
        data = json.load(f)
    logger.info("Loaded %d samples from %s", len(data), result_json_path)

    scores = llm_as_judge(data)
    if not scores:
        raise ValueError("No scores were generated. Please check the input data and model configuration.")

    valid_scores = [s for s in scores if s != -1]
    verdicts = [score_to_verdict(s) for s in valid_scores]

    avg_all_scores = sum(scores) / len(scores) if scores else 0
    avg_valid_scores = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    accuracy = sum(verdicts) / len(verdicts) if verdicts else 0

    logger.info("num_samples=%d num_valid=%d", len(scores), len(valid_scores))
    logger.info("avg_all_scores=%.4f avg_valid_scores=%.4f accuracy=%.4f",
                avg_all_scores, avg_valid_scores, accuracy)

    per_sample = pd.DataFrame({
        'qid': [s.get('qid') for s in data],
        'question': [s['question'] for s in data],
        'gt_answer': [s['gt_answer'] for s in data],
        'generated_answer': [s['generated_answer'] for s in data],
        'score': scores,
        'verdict': [score_to_verdict(s) for s in scores],
    })
    per_sample.to_csv(osp.join(experiment_dir, 'evaluation_results.csv'), index=False)

    results = {
        'accuracy': accuracy,
        'avg_all_scores': avg_all_scores,
        'avg_valid_scores': avg_valid_scores,
        'num_samples': len(scores),
        'num_valid_samples': len(valid_scores),
    }
    with open(osp.join(experiment_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    metadata['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    metadata['duration_seconds'] = round(time.time() - start_perf, 2)
    metadata['num_samples'] = len(scores)
    metadata['num_valid_samples'] = len(valid_scores)
    with open(osp.join(experiment_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Wrote results to %s", experiment_dir)


