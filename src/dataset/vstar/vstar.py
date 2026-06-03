import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm/")

import os

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
# from src.utils.vision_process import fetch_video # enforces 16 frames by default, but doing that would disorient the annotations overlay (training on 16 frames though)
from qwen_vl_utils import fetch_video

from src.utils.annotation_utils import load_mask_annotations, annotate_frames


class VStarDataset(Dataset):
    """Dataset for V-STaR visual grounding benchmark.

    Args:
        dataset_root: Root directory of the V-STaR dataset.
        system_prompt_path: Path to a text file containing the system prompt.
        sam3_annotations_root: Root directory of SAM3 annotations.
        annotate_concepts: If True, draw bounding boxes on video frames.
        bbox_thickness: Pixel thickness of drawn bounding boxes.
        annotate_frame_numbers: If True, overlay each frame's index at the bottom-right corner.
        frame_number_scale: Font scale for the frame number overlay.
        frame_number_thickness: Font thickness for the frame number overlay.
        video_config: Dict of video processing params (e.g. total_pixels, fps) passed into the video message.
    """

    def __init__(
        self,
        dataset_root: str,
        system_prompt_path: str,
        sam3_annotations_root: str = None,
        video_config: dict[str, str | int] | None = None,
        annotate_concepts: bool = False,
        bbox_thickness: int = 2,
        annotate_frame_numbers: bool = False,
        frame_number_scale: float = 1.0,
        frame_number_thickness: int = 2,
    ):
        self.dataset_root = dataset_root
        self.video_dir = os.path.join(dataset_root, "videos")
        self.annotate_concepts = annotate_concepts
        self.bbox_thickness = bbox_thickness
        self.annotate_frame_numbers = annotate_frame_numbers
        self.frame_number_scale = frame_number_scale
        self.frame_number_thickness = frame_number_thickness
        self.sam3_annotations_root = os.path.join(sam3_annotations_root, "masks")
        self.video_config = video_config or {}

        annotation_path = os.path.join(dataset_root, "V_STaR_test_with_qid.json")
        hf_dataset = load_dataset("json", data_files=annotation_path, split="train")
        self.samples = list(hf_dataset)

        with open(system_prompt_path) as f:
            self.system_prompt = f.read().strip()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = os.path.join(self.video_dir, sample["vid"] + ".mp4")

        frames = self._load_video_frames(video_path, sample['qid'])
        frame_prompt = "".join(
            f"Frame {i}: <|vision_start|><|image_pad|><|vision_end|>\n"
            for i in range(len(frames))
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["question"]},
                    {"type": "video"},
                ],
            },
        ]

        return {
            "messages": messages,
            "frames": frames,
            "frame_prompt": frame_prompt,
            "qid": sample["qid"],
            "vid": sample["vid"],
            "domain": sample.get("domain"),
            "video_path": video_path,
            "question": sample["question"],
            "answer": sample.get("answer"),
        }

    def _load_video_frames(self, video_path: str, qid: str) -> torch.Tensor:
        """Load video frames as a tensor.

        Args:
            video_path: Path to the video file.
            qid: Question ID used to locate annotation file.
        Returns:
            Tensor of shape (n, c, h, w) with pixel values in [0, 255].
        """
        video_config = self.video_config.copy()
        video_config['video'] = video_path
        video_input = fetch_video(video_config)  # (n, c, h, w), values in [0, 255]

        if self.annotate_concepts:
            annotation_path = os.path.join(self.sam3_annotations_root, f"{qid}.json")
            if os.path.exists(annotation_path):
                record = load_mask_annotations(annotation_path)
                annotated = annotate_frames(video_input, record["masks_np"], annotation_type="bbox", thickness=self.bbox_thickness)
                video_input = torch.tensor(np.array(annotated)).permute(0, 3, 1, 2).to(torch.float32)

        if self.annotate_frame_numbers:
            video_input = self._annotate_frame_numbers(video_input)

        return video_input

    def _annotate_frame_numbers(self, frames: torch.Tensor) -> torch.Tensor:
        """Overlay each frame's index at the bottom-right corner.

        Args:
            frames: Tensor of shape (n, c, h, w).

        Returns:
            Tensor of shape (n, c, h, w) with frame indices overlaid.
        """
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
