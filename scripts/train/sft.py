# adapted from: Open-o3-Video/src/r1-v/src/open_r1/sft_multi_task.py
import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
import torch

from datasets import load_dataset
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

from src.utils.vision_process import process_vision_info

import logging
logging.basicConfig(level=logging.INFO)
os.environ["DECORD_LOG_LEVEL"] = "ERROR"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("src.utils.vision_process").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class MyScriptArguments(ScriptArguments):
    dataset_root: str = field(
        default="/projects/torresani-lab/ajay/datasets/Open-o3-Video/",
        metadata={"help": "Root directory for the dataset that houses the videos/ folder."},
    )
    wandb_project_name: str = field(
        default="huggingface",
        metadata={"help": "Wandb project name."},
    )
    system_prompt_path: str = field(
        default="prompts/system_prompt.txt",
        metadata={"help": "Path to system prompt text file."},
    )

def get_video_path(example: dict[str, any], data_root: str) -> str:

    task = example["task"]
    source = example.get("source", "")
    root = os.path.join(data_root, "videos")

    if task == "visual QA":
        video_root = os.path.join(root, "treevgr")
        path = os.path.join(video_root, example["image_path"])
    elif task == "temporal-spatial free-form QA":
        if source == "STR_plm_rdcap":
            video_root = os.path.join(root, "stgr/plm/videos")
        else:
            video_root = os.path.join(root, "stgr/temporal_grounding/videos")
        path = os.path.join(video_root, example["video_path"])
    elif task == "temporal QA":
        path = os.path.join(root, "tvg_r1", example["video_path"])
    elif "General video QA" in task:
        path = os.path.join(root, "videor1", example["video_path"])
    else:
        raise ValueError(f"Unknown task: {task}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    return path

def get_key_frame_path(key_frame: dict[str, any],
                       example: dict[str, any],
                       data_root: str) -> str:
    root = os.path.join(data_root, "videos")

    if example.get("source") == "STR_plm_rdcap":
        key_frame_root = os.path.join(root, "stgr/plm/kfs")
    else:
        key_frame_root = os.path.join(root, "stgr/temporal_grounding/kfs")

    path = os.path.join(key_frame_root, key_frame["path"])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    return path

def prepare_dataset(dataset, data_root: str, system_prompt_path: str):
    """collator expects each item to have messages, key_frames, task"""

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read().strip()

    new_dataset = []
    for example in dataset:

        if example["task"] != "temporal-spatial free-form QA":
            continue

        reasoning = example['reasoning_process'].replace("<think>", "").replace("</think>", "").strip()
        assistant_text = f"<think>{reasoning}</think>\n<answer>{example['answer']}</answer>"

        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system_prompt,
                }]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": example['question'],
                    },
                    {
                        "type": "video",
                        "video": get_video_path(example, data_root)
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": assistant_text,
                }]
            },
        ]

        key_frames = [
            {**kf, "path": get_key_frame_path(kf, example, data_root)}
            for kf in example["key_frames"]
        ]

        new_dataset.append({"messages": messages, 
                            "key_frames": key_frames, 
                            "task": example['task'], 
                            "source": example["source"], 
                            "image_size": []})
    
    return new_dataset

def custom_collate_fn(examples: list[dict[str, any]]):

    assert(len(examples) == 1), "Batch size > 1 not supported for the custom collate function."
    example = examples[0]   
    assert(example["task"] == "temporal-spatial free-form QA"), f"Unsupported task: {example['task']} | Supported task(s): temporal-spatial free-form QA"
    
    # process vision part of input
    _, video_inputs, video_kwargs = process_vision_info(example['messages'], return_video_kwargs=True)
    _, _, h, w = video_inputs[0].shape # n, c, h, w

    # load key_frames
    key_frames = []
    for kf in example['key_frames']:
        kf_time = kf['time']
        kf_tensor = Image.open(kf['path']).convert("RGB").resize((w, h))
        kf_tensor = torch.from_numpy(np.array(kf_tensor)).permute(2, 0, 1) # c, h, w
        key_frames.append((kf_time, kf_tensor))

    # insert key frames into video inputs at appropriate time steps
    curr_kf_idx, sampled_fps = 0, video_kwargs['fps'][0]
    merged_video_inputs = []

    for idx, frame in enumerate(video_inputs[0]):
        # find current time
        curr_time = int(idx / sampled_fps)
        # insert all kfs that have occured up until now
        while curr_kf_idx < len(key_frames) and key_frames[curr_kf_idx][0] <= curr_time:
            merged_video_inputs.append(key_frames[curr_kf_idx])
            curr_kf_idx += 1
        # insert current frame
        merged_video_inputs.append((round(idx / sampled_fps, 1), frame))

    # Add frame prompts to make the model temporally aware of the provided input frames
    frame_prompt = "".join(
        f"Frame {i+1} at {t}s: <|vision_start|><|image_pad|><|vision_end|>\n"
        for i, (t, _) in enumerate(merged_video_inputs)
    )

    # drop the time information 
    refined_video_inputs = torch.stack([frame for _, frame in merged_video_inputs], dim=0) # n, c, h, w
    
    # process textual part of input
    processed_text = processor.apply_chat_template(example['messages'], tokenize=False)
    texts = [processed_text.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)]

    output = processor(
        text=texts,
        images=[refined_video_inputs], # as we restrict batch size to 1
        videos=None, # as we already broke down the video and have a frame prompt that references images now
        return_tensors="pt",
        padding=True, # essentially a no-op since batch size is 1
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
    parser = TrlParser((MyScriptArguments, SFTConfig, ModelConfig))
    script_args, sft_config, model_config = parser.parse_args_and_config()
    os.environ["WANDB_PROJECT"] = script_args.wandb_project_name

    sft_config.gradient_checkpointing_kwargs = {"use_reentrant": False} # to prevent issues during PEFT training
    sft_config.remove_unused_columns = False # to prevent unused columns from being removed 
    sft_config.dataset_kwargs = {"skip_prepare_dataset": True} # since we are using a custom collator

    # Load dataset
    dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")
    dataset = prepare_dataset(dataset, script_args.dataset_root, script_args.system_prompt_path)
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