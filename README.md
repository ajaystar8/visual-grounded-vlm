# Do Video VLMs Use Grounded Visual Evidence?

This repository documents a research project I worked on at the FarSight Lab at Northeastern University with Prof. Lorenzo Torresani (November 2025 to June 2026).

I began with a simple question: **if a video vision-language model is given explicit evidence about which regions matter for a question, does it actually use that evidence in its reasoning?** I explored three ways of providing that guidance: textual descriptions of grounded regions, visual overlays, and grounded chain-of-thought supervision.

The answer was more sobering than I expected. The models often learned to *sound* grounded without becoming reliably dependent on the evidence they cited. They could mention boxes, regions, and tracked objects while ignoring incorrect evidence or inventing visual details that were not present.

This work did not become a paper, but I did not want the experiments, infrastructure, and lessons from it to disappear. I am making the code public as an honest record of the research process: the ideas that worked, the ones that stalled, and the evidence that changed how I thought about grounded reasoning in VLMs.

## At a glance

- **Research question:** Can external, spatially grounded evidence make video-VLM reasoning more faithful?
- **Main finding:** Models frequently adopted the language of grounding without consistently using the grounded regions to determine their answers.
- **Benchmark:** [V-STaR](https://github.com/V-STaR-Bench/V-STaR) for spatio-temporal video question answering.
- **Training data:** The STR split of [Open-o3 Video](https://github.com/marinero4972/Open-o3-Video).
- **Models:** Qwen2.5-VL-7B as the model under study; Qwen3-VL-32B and Perception-LM-8B for caption and reasoning-trace generation; SAM3 for grounding and tracking; and Qwen2.5-72B/Qwen3-4B as LLM judges.

## What I tried and what I learned

### 1. Describing grounded regions as text

I first used SAM3 to ground question-relevant entities and track them through each video. A VLM then produced a region caption (RCAP) for every tracked entity. I supplied those captions to the model alongside the video and, as a text-only control, without the video.

The added evidence made the model abstain more often, but it rarely corrected an answer that was already wrong. Controlled experiments with accurate and inaccurate captions suggested that even accurate region evidence was seldom integrated into the model's reasoning.

**Implementation:** [SAM3 grounding and tracking](src/dataset/openo3/sam3_track.py) · [Perception-LM region captions](src/dataset/openo3/plm_rcap.py) · [V-STaR region captions](src/dataset/vstar/rcap.py) · [inference with captions and video](scripts/benchmark/vstar/qwen_rcap_infer_vg.py) · [text-only control](scripts/benchmark/vstar/qwen_rcap_infer_blind.py)

### 2. Drawing the evidence on the video

Next, I propagated SAM3 masks from V-STaR's ground-truth boxes to every sampled frame and rendered the result as boxes, masks, or outlines. I evaluated these overlays zero-shot and after LoRA fine-tuning on Open-o3 keyframes annotated with red boxes.

The overlays improved VQA performance, which led me to explore a question-conditioned grounding module trained with GRPO. That direction ultimately stalled for an important reason: the model's answers barely changed when the overlaid evidence was deliberately made wrong. Without a meaningful behavioral response to the evidence, the reward model had little signal from which to learn.

**Implementation:** [mask propagation](src/dataset/vstar/propagate_sam3_masks.py) · [overlay utilities](src/utils/annotation_utils.py) · [inference with overlays](scripts/benchmark/vstar/qwen_infer_vg.py) ([box, mask, and outline configs](config/infer/qwen25vl7b_vstar_zeroshot_vg/)) · [keyframe-box dataset](src/dataset/openo3/stqa.py) · [keyframe-box SFT](scripts/train/sft_bbox.py) · [LLM-judge evaluation](scripts/benchmark/vstar/eval.py), which scores each answer from 0 to 3 against the reference

### 3. Supervising a grounded chain of thought

Finally, I designed a reasoning template that asks the model to inspect each boxed entity, follow it across frames, and only then produce an answer. The data-generation pipeline has four stages:

1. Qwen3-VL-32B captions each tracked entity.
2. The model rewrites those captions as a reasoning trace that follows the grounding template.
3. An answer-overlap heuristic and a Qwen3-4B judge filter broken traces.
4. The surviving traces are used to LoRA fine-tune Qwen2.5-VL-7B on densely boxed frames.

After fine-tuning, the model did begin to cite the boxes. However, it also cited boxes that had never been drawn and incorrectly described their contents. The reasoning trace occasionally overturned a wrong answer, but fabricated evidence was much more common. My central takeaway was that **teaching a model the form of grounded reasoning is not the same as making its answer causally depend on grounded evidence**.

**Implementation:** [CoT template](prompts/cot/v15_only_rcap_estb_rln_no_gt_calib_ans.md) ([earlier iterations](prompts/cot/)) · [trace generation](src/dataset/openo3/str_cot_synth.py) → [verification and filtering](src/dataset/openo3/cot_verify_refine.py) → [dense-box dataset](src/dataset/openo3/openo3_video_dataset.py) → [CoT SFT](scripts/train/sft_cot.py) · [inference](scripts/benchmark/vstar/qwen_infer_vg_dense.py)

**Analysis:** [CoT trace browser](notebooks/sft_cot_browser.ipynb) (manual inspection, looping behavior, and filtering statistics) · [attention maps](notebooks/attention_map_analysis.ipynb) (answer-to-frame attention with and without overlays)

## Repository map

```text
src/dataset/    dataset classes and data-generation pipelines for V-STaR and Open-o3
src/utils/      frame sampling, mask/bounding-box overlays, and Open-o3 helpers
scripts/        training, V-STaR inference and evaluation, and vLLM client notebooks
config/         YAML configurations organized by dataset, inference, training, and tracking
prompts/        versioned prompts for captioning, inference, training, and grounded CoT
bash_scripts/   Slurm launchers that mirror the experiment structure above
notebooks/      analysis notebooks with saved outputs
```

I kept the prompt iterations and experiment-specific configurations because they show how the approach evolved and make the main runs traceable. The repository is a research artifact rather than a polished library.

## Reproducing the experiments

These experiments were run on a Slurm cluster. Dataset, checkpoint, environment, and output paths in the YAML and `sbatch` files still refer to that cluster, so they must be updated for a different system. The main dependencies are `vllm`, `transformers`, `trl`, `peft`, `sam3`, `torchcodec`, `qwen-vl-utils`, and `pycocotools`.

Run all commands from the repository root. Create the log directory before submitting jobs:

```bash
mkdir -p slurm_logs

# Build pseudo-ground-truth masks for V-STaR, run overlay-guided inference,
# and score the resulting answers with an LLM judge.
sbatch bash_scripts/dataset/propagate_sam3_masks_vstar.sh
sbatch bash_scripts/benchmark/vstar/qwen_infer_vg.sh
sbatch bash_scripts/benchmark/vstar/eval.sh

# Build grounded-CoT data, then fine-tune on the verified traces.
sbatch bash_scripts/track/openo3_str_sam3_track.sh
sbatch bash_scripts/dataset/openo3_str_cot_synth.sh
sbatch bash_scripts/dataset/cot_verify_refine.sh
sbatch bash_scripts/train/sft.sh
```

Most Python entry points accept `--config <path-to-yaml>`. The `sbatch` launchers select the corresponding configuration; for training, `SFT_SCRIPT` and `CONFIG_FILE` can be set to choose a different run.

## Scope and limitations

This code reflects an exploratory research workflow rather than a maintained software package. It assumes access to the original datasets, model checkpoints, and a GPU/Slurm environment; it has not been packaged as a turnkey reproduction. I have tried to preserve the configurations, prompt history, and analysis artifacts needed to understand each stage and adapt it to another system.

The broader lesson I took from this project is that output which appears grounded is not sufficient evidence of grounded reasoning. A useful evaluation should test whether the model's behavior changes when the supplied evidence changes, not only whether its explanation mentions the right regions. That lesson is what I most wanted to preserve from this work.
