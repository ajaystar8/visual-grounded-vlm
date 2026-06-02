import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from src.utils.vision_process import process_vision_info


@dataclass
class OpenO3DatasetConfig:
    annotation_json_path: str = field(
        metadata={"help": "Path to the STGR-SFT JSON file."},
    )
    dataset_root: str = field(
        default="/projects/torresani-lab/ajay/datasets/Open-o3-Video/",
        metadata={"help": "Root directory housing the videos/ folder."},
    )
    system_prompt_path: str = field(
        default="prompts/train/openo3_bbox_stqa.md",
        metadata={"help": "Path to system prompt text file."},
    )
    annotate_concepts: bool = field(
        default=False,
        metadata={"help": "Draw bounding boxes on key frames before merging."},
    )
    replace_boxes: bool = field(
        default=False,
        metadata={"help": "Replace <box>[...]</box> tokens in reasoning with replacement_text."},
    )
    replacement_text: str = field(
        default="as highlighted by the red bounding box",
        metadata={"help": "Text to substitute for <box> tokens when replace_boxes is True."},
    )
    bbox_thickness: int = field(
        default=2,
        metadata={"help": "Pixel thickness of bounding boxes drawn on key frames."},
    )
    annotate_frame_numbers: bool = field(
        default=False,
        metadata={"help": "Overlay frame index at bottom-right corner of each merged frame."},
    )
    frame_number_scale: float = field(
        default=1.0,
        metadata={"help": "Font scale for the frame number overlay."},
    )
    frame_number_thickness: int = field(
        default=2,
        metadata={"help": "Font thickness for the frame number overlay."},
    )


class OpenO3STQADataset(Dataset):
    """Dataset for Open-o3-Video temporal-spatial free-form QA task.

    Args:
        json_path: Path to the STGR-SFT JSON file.
        data_root: Root directory housing the videos/ folder.
        system_prompt_path: Path to a text file containing the system prompt.
        annotate_concepts: If True, draw bounding boxes on key frames before merging.
        replace_boxes: If True, replace <box>[...]</box> tokens in reasoning with replacement_text.
        replacement_text: Text to substitute when replace_boxes is True.
        bbox_thickness: Pixel thickness of drawn bounding boxes when annotate_frames is True.
        annotate_frame_numbers: If True, overlay each frame's index at the bottom-right corner.
        frame_number_scale: Font scale for the frame number overlay.
        frame_number_thickness: Font thickness for the frame number overlay.
    """

    TASK = "temporal-spatial free-form QA"

    @classmethod
    def from_config(cls, config: "OpenO3DatasetConfig") -> "OpenO3STQADataset":
        return cls(
            json_path=config.annotation_json_path,
            data_root=config.dataset_root,
            system_prompt_path=config.system_prompt_path,
            annotate_concepts=config.annotate_concepts,
            replace_boxes=config.replace_boxes,
            replacement_text=config.replacement_text,
            bbox_thickness=config.bbox_thickness,
            annotate_frame_numbers=config.annotate_frame_numbers,
            frame_number_scale=config.frame_number_scale,
            frame_number_thickness=config.frame_number_thickness,
        )

    def __init__(
        self,
        json_path: str,
        data_root: str,
        system_prompt_path: str,
        annotate_concepts: bool = False,
        replace_boxes: bool = False,
        replacement_text: str = "as highlighted by the red bounding box",
        bbox_thickness: int = 2,
        annotate_frame_numbers: bool = False,
        frame_number_scale: float = 1.0,
        frame_number_thickness: int = 2,
    ):
        self.data_root = data_root
        with open(system_prompt_path, "r") as f:
            self.system_prompt = f.read().strip()
        self.annotate_concepts = annotate_concepts
        self.replace_boxes = replace_boxes
        self.replacement_text = " " + replacement_text.strip() + " " # add spaces for better token separation
        self.bbox_thickness = bbox_thickness
        self.annotate_frame_numbers = annotate_frame_numbers
        self.frame_number_scale = frame_number_scale
        self.frame_number_thickness = frame_number_thickness

        hf_dataset = load_dataset("json", data_files=json_path, split="train")
        self.samples = [ex for ex in hf_dataset if ex["task"] == self.TASK]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example["question"]},
                    {"type": "video", "video": self._get_video_path(example)},
                ],
            },
            # assistant message to follow, keep reading
        ]

        key_frame_paths = sorted(
            [{**kf, "path": self._get_key_frame_path(kf, example)} for kf in example["key_frames"]],
            key=lambda kf: kf["time"],
        )

        merged_frames, frame_prompt, key_frame_indices, key_frame_bboxes = self._merge_key_frames(messages, key_frame_paths, example)

        # build time -> merged frame index map from key frames sorted by time
        kf_times_sorted = sorted(kf["time"] for kf in example["key_frames"])
        time_to_merged_idx = {t: key_frame_indices[i] for i, t in enumerate(kf_times_sorted)}

        # preprocess CoT trace
        # replace <box> tags with replacement text
        reasoning = example["reasoning_process"].replace("<think>", "").replace("</think>", "").strip()
        if self.replace_boxes:
            reasoning = re.sub(r"<box>\[.*?\]</box>", self.replacement_text, reasoning)

        # replace <t> tags with corresponding frame placeholders based on time_to_merged_idx mapping
        def replace_time_tag(m):
            t = float(m.group(1))
            merged_idx = time_to_merged_idx.get(t)
            return f" <frame>{merged_idx}</frame>" if merged_idx is not None else m.group(0)
        reasoning = re.sub(r"<t>([\d.]+)</t>s?", replace_time_tag, reasoning)
        assistant_text = f"<think>{reasoning}</think>\n<answer>{example['answer']}</answer>"

        # append assistant message 
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        })

        return {"messages": messages,
                "merged_frames": merged_frames,
                "frame_prompt": frame_prompt,
                "key_frame_indices": key_frame_indices,
                "key_frame_bboxes": key_frame_bboxes}

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _get_video_path(self, example: dict) -> str:
        task = example["task"]
        source = example.get("source", "")
        root = os.path.join(self.data_root, "videos")

        if task == "visual QA":
            path = os.path.join(root, "treevgr", example["image_path"])
        elif task == self.TASK:
            subdir = "stgr/plm/videos" if source == "STR_plm_rdcap" else "stgr/temporal_grounding/videos"
            path = os.path.join(root, subdir, example["video_path"])
        elif task == "temporal QA":
            path = os.path.join(root, "tvg_r1", example["video_path"])
        elif "General video QA" in task:
            path = os.path.join(root, "videor1", example["video_path"])
        else:
            raise ValueError(f"Unknown task: {task}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path

    def _get_key_frame_path(self, key_frame: dict, example: dict) -> str:
        root = os.path.join(self.data_root, "videos")
        subdir = "stgr/plm/kfs" if example.get("source") == "STR_plm_rdcap" else "stgr/temporal_grounding/kfs"
        path = os.path.join(root, subdir, key_frame["path"])

        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path

    # ------------------------------------------------------------------
    # Frame merging
    # ------------------------------------------------------------------

    def _merge_key_frames(self, messages: list, key_frame_paths: list, example: dict) -> tuple[torch.Tensor, str, list[int], list[dict]]:
        """Merge annotated key frames into the sampled video frames.

        Returns:
            merged_frames: Tensor of shape (n, c, h, w).
            frame_prompt: Text with per-frame image placeholders.
            key_frame_indices: Indices into merged_frames where key frames were inserted.
            key_frame_bboxes: List parallel to key_frame_indices; each entry is the
                {label: [bbox, ...]} mapping (normalized xyxy) for that key frame.
        """
        _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        _, _, h, w = video_inputs[0].shape

        key_frames = []
        for kf in key_frame_paths:
            img = Image.open(kf["path"]).convert("RGB").resize((w, h))
            if self.annotate_concepts:
                img = self._draw_boxes(img, kf, example, (w, h))
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            bboxes = example.get("key_items", {}).get(str(kf["idx"]), {})
            key_frames.append((kf["time"], tensor, bboxes))

        curr_kf_idx, sampled_fps = 0, video_kwargs["fps"][0]
        merged = []
        key_frame_indices = []
        key_frame_bboxes = []

        for idx, frame in enumerate(video_inputs[0]):
            curr_time = int(idx / sampled_fps)
            while curr_kf_idx < len(key_frames) and key_frames[curr_kf_idx][0] <= curr_time:
                key_frame_indices.append(len(merged))
                kf_time, kf_tensor, kf_bboxes = key_frames[curr_kf_idx]
                merged.append((kf_time, kf_tensor))
                key_frame_bboxes.append(kf_bboxes)
                curr_kf_idx += 1
            merged.append((round(idx / sampled_fps, 1), frame))

        # flush key frames whose time exceeds the last sampled video frame
        while curr_kf_idx < len(key_frames):
            key_frame_indices.append(len(merged))
            kf_time, kf_tensor, kf_bboxes = key_frames[curr_kf_idx]
            merged.append((kf_time, kf_tensor))
            key_frame_bboxes.append(kf_bboxes)
            curr_kf_idx += 1

        # use specific time stamps in frame prompt
        # frame_prompt = "".join(
        #     f"Frame {i+1} at {t}s: <|vision_start|><|image_pad|><|vision_end|>\n"
        #     for i, (t, _) in enumerate(merged)
        # )
        # use frame numbers in frame prompt
        frame_prompt = "".join(
            f"Frame {i}: <|vision_start|><|image_pad|><|vision_end|>\n"
            for i, _ in enumerate(merged)
        )
        merged_frames = torch.stack([f for _, f in merged], dim=0)

        if self.annotate_frame_numbers:
            merged_frames = self._annotate_frame_numbers(merged_frames)

        return merged_frames, frame_prompt, key_frame_indices, key_frame_bboxes

    def _draw_boxes(self, img: Image.Image, kf: dict, example: dict, image_size: tuple) -> Image.Image:
        """Draw bounding boxes for all items in this key frame onto img."""
        import cv2

        idx_str = str(kf["idx"])
        items = example.get("key_items", {}).get(idx_str, {})
        if not items:
            return img

        w, h = image_size
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        for label, bboxes in items.items():
            for bbox in bboxes:
                x1 = int(round(bbox[0] * w))
                y1 = int(round(bbox[1] * h))
                x2 = int(round(bbox[2] * w))
                y2 = int(round(bbox[3] * h))
                cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 0, 255), self.bbox_thickness)

        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    def _annotate_frame_numbers(self, frames: torch.Tensor) -> torch.Tensor:
        """Overlay each frame's index at the bottom-right corner."""
        annotated = []
        for i, frame in enumerate(frames):
            arr = cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
            h, w = arr.shape[:2]
            text = str(i)
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, self.frame_number_scale, self.frame_number_thickness
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            x, y = w - tw - 8, h - baseline - 4
            cv2.putText(arr, text, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
            t = torch.from_numpy(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            annotated.append(t)
        return torch.stack(annotated)
