# adapted from: Open-o3-Video/src/r1-v/src/open_r1/sft_multi_task.py
import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import os.path as osp
os.environ["WANDB_PROJECT"] = "visual-grounded-vlm"

import dataclasses
import yaml
from datetime import datetime
import torch
from torchvision.transforms.functional import to_pil_image as to_pil

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import (
    TrlParser, 
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
)
from peft import get_peft_model

from src.dataset.openo3.openo3_video_dataset import OpenO3VideoDataset, OpenO3VideoDatasetConfig

import logging
logging.basicConfig(level=logging.INFO)
os.environ["DECORD_LOG_LEVEL"] = "ERROR"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("src.utils.vision_process").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def save_run_config(output_dir: str, *configs) -> str:
    """Save all dataclass configs to a single YAML file in output_dir, with system prompt contents inlined."""
    os.makedirs(output_dir, exist_ok=True)
    path = osp.join(output_dir, "run_config.yaml")
    with open(path, "w") as f:
        f.write(f"# timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for cfg in configs:
            f.write(f"# {type(cfg).__name__}\n")
            yaml.dump(dataclasses.asdict(cfg), f, default_flow_style=False)
            f.write("\n")
    return path

def build_messages(sample: dict, system_prompt: str) -> tuple[list[dict], list]:
    # convert annotated frames to a list of PIL images (one image content block each)
    images = [to_pil(frame) for frame in sample["annotated_frames"]]

    # wrap the reasoning preceding <answer> in <think>...</think>, keeping the answer tag
    reasoning, _, answer = sample["gt_cot"]["cot"].partition("<answer>")
    target = f"<think>{reasoning.strip()}</think>\n<answer>{answer}".strip()

    # build the chat: system prompt, user (annotated frames + question), assistant (think+answer target)
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in images],
                {"type": "text", "text": f"Question: {sample['question']}"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": target},
            ],
        },
    ]
    return messages, images


def custom_collate_fn(examples: list[dict[str, any]]):

    assert len(examples) == 1, "Batch size > 1 not supported for the custom collate function."
    example = examples[0]

    messages, images = build_messages(example, system_prompt)
    processed_text = [processor.apply_chat_template(messages, tokenize=False)]

    output = processor(
        text= processed_text,
        images=images,
        videos=None,
        return_tensors="pt",
        padding=True,
        do_resize=False,
    )

    labels = output["input_ids"].clone()
    input_ids = output["input_ids"][0]

    # mask system + user tokens: only compute loss on the assistant turn
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_ids = processor.tokenizer.encode("assistant", add_special_tokens=False)
    assistant_start = None
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i] == im_start_id and input_ids[i+1:i+1+len(assistant_ids)].tolist() == assistant_ids:
            assistant_start = i
            break
    if assistant_start is not None:
        labels[0, :assistant_start] = -100

    # mask padding tokens and vision tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    visual_tokens = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    if not isinstance(visual_tokens, list):
        visual_tokens = [visual_tokens]
    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    output["labels"] = labels
    return output


if __name__ == "__main__":

    # Setup parsers and configs
    parser = TrlParser((OpenO3VideoDatasetConfig, SFTConfig, ModelConfig))
    dataset_config, sft_config, model_config = parser.parse_args_and_config()

    sft_config.output_dir = osp.join(sft_config.output_dir, sft_config.run_name)
    if os.path.exists(sft_config.output_dir):
        raise FileExistsError(f"Output directory already exists: {sft_config.output_dir}. Delete it or use a different run_name.")
    run_config_path = save_run_config(sft_config.output_dir, dataset_config, sft_config, model_config)
    logger.info(f"Saved run config to {run_config_path}")

    # load the system prompt template used to frame each training turn
    with open(dataset_config.sft_system_prompt_path) as f:
        system_prompt = f.read()

    sft_config.gradient_checkpointing_kwargs = {"use_reentrant": False} # to prevent issues during PEFT training
    sft_config.remove_unused_columns = False # to prevent unused columns from being removed 
    sft_config.dataset_kwargs = {"skip_prepare_dataset": True} # since we are using a custom collator

    # Load dataset
    dataset = OpenO3VideoDataset.from_config(dataset_config)
    logging.info(f"Loaded dataset with {len(dataset)} samples belonging to `temporal-spatial free-form QA` task.")

    # Create model instance
    torch_dtype = (
        model_config.dtype
        if model_config.dtype in ["auto", None]
        else getattr(torch, model_config.dtype)
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=get_kbit_device_map()
    )
    logging.info(f"Loaded model '{model_config.model_name_or_path}' with dtype {torch_dtype}.")

    # Apply PEFT configuration to the model
    if model_config.use_peft:
        lora_config = get_peft_config(model_config)
        model = get_peft_model(model, lora_config)
        logging.info(f"Applied PEFT configuration to model.")
        model.print_trainable_parameters()
    else: 
        logging.info("PEFT not enabled. Training the full model.")

    # Create processor instance
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        data_collator=custom_collate_fn, 
    )

    # Start training
    trainer.train()

    # Checkpointing model
    trainer.save_model(sft_config.output_dir)

    # Checkpointing processor
    processor.save_pretrained(sft_config.output_dir)

    # Re-enable KV cache if it was disabled for training
    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(sft_config.output_dir)

    # Cleanup
    del trainer
    del model
    del processor
    torch.cuda.empty_cache()
    if "wandb" in sft_config.report_to:
        import wandb
        wandb.finish()