# adapted from: Open-o3-Video/src/r1-v/src/open_r1/sft_multi_task.py
import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
import dataclasses
import yaml
from datetime import datetime
from dataclasses import dataclass, field

import torch

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import (
    TrlParser, 
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
)
from peft import get_peft_model

from src.dataset.openo3.stqa import OpenO3STQADataset, OpenO3DatasetConfig

import logging

def save_run_config(output_dir: str, dataset_config: "OpenO3DatasetConfig", *configs) -> str:
    """Save all dataclass configs to a single YAML file in output_dir, with system prompt contents inlined."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_config.yaml")
    with open(path, "w") as f:
        f.write(f"# timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for cfg in (dataset_config, *configs):
            f.write(f"# {type(cfg).__name__}\n")
            d = dataclasses.asdict(cfg)
            if cfg is dataset_config and os.path.isfile(dataset_config.system_prompt_path):
                with open(dataset_config.system_prompt_path) as sp:
                    d["system_prompt_contents"] = sp.read()
            yaml.dump(d, f, default_flow_style=False)
            f.write("\n")
    return path


logging.basicConfig(level=logging.INFO)
os.environ["DECORD_LOG_LEVEL"] = "ERROR"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("src.utils.vision_process").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class MyScriptArguments(ScriptArguments):
    wandb_project_name: str = field(
        default="huggingface",
        metadata={"help": "Wandb project name."},
    )


def custom_collate_fn(examples: list[dict[str, any]]):

    assert len(examples) == 1, "Batch size > 1 not supported for the custom collate function."
    example = examples[0]

    processed_text = processor.apply_chat_template(example['messages'], tokenize=False)
    texts = [processed_text.replace("<|vision_start|><|video_pad|><|vision_end|>", example['frame_prompt'])]

    output = processor(
        text=texts,
        images=[example['merged_frames']],
        videos=None,
        return_tensors="pt",
        padding=True,
        do_resize=False,
    )

    labels = output["input_ids"].clone()

    # do not compute loss on padding tokens and vision tokens (adapted from trl/trainer/sft_trainer.py)
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
    parser = TrlParser((MyScriptArguments, OpenO3DatasetConfig, SFTConfig, ModelConfig))
    script_args, dataset_config, sft_config, model_config = parser.parse_args_and_config()
    os.environ["WANDB_PROJECT"] = script_args.wandb_project_name

    sft_config.output_dir = os.path.join(sft_config.output_dir, sft_config.run_name)
    if os.path.exists(sft_config.output_dir):
        raise FileExistsError(f"Output directory already exists: {sft_config.output_dir}. Delete it or use a different run_name.")
    run_config_path = save_run_config(sft_config.output_dir, dataset_config, script_args, sft_config, model_config)
    logger.info(f"Saved run config to {run_config_path}")

    sft_config.gradient_checkpointing_kwargs = {"use_reentrant": False} # to prevent issues during PEFT training
    sft_config.remove_unused_columns = False # to prevent unused columns from being removed 
    sft_config.dataset_kwargs = {"skip_prepare_dataset": True} # since we are using a custom collator

    # Load dataset
    dataset = OpenO3STQADataset(
        json_path=dataset_config.annotation_json_path,
        data_root=dataset_config.dataset_root,
        system_prompt_path=dataset_config.system_prompt_path,
        annotate_frames=dataset_config.annotate_frames,
        replace_boxes=dataset_config.replace_boxes,
        replacement_text=dataset_config.replacement_text,
        bbox_thickness=dataset_config.bbox_thickness,
    )
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