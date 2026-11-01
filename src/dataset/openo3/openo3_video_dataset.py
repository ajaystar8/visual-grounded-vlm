import sys
sys.path.append("/projects/torresani-lab/ajay/visual-grounded-vlm")

import os
import json
from glob import glob
from dataclasses import dataclass, field

import cv2
import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from qwen_vl_utils import fetch_video

from src.utils.openo3_utils import (
    get_video_path, splice_keyframes,
    load_track_manifest, load_track_masks, masks_to_bboxes,
)

GT_ANNOTATION_PATH = "/projects/torresani-lab/ajay/datasets/Open-o3-Video/json_data/STGR-SFT.json"
SAM3_TRACKS_PATH = "/scratch/rajendrakumar.a/visual-grounded-vlm/annotations/openo3video/uniform_sample_1fps_orig_res_track"
COT_PATH = "/scratch/rajendrakumar.a/visual-grounded-vlm/annotations/openo3video/openo3_str_cot_synth_no_gt_ans"

BBOX_ANNOTATION_CONFIG = {
    "color": (255, 0, 0),
    "thickness": 2
}
FPS = 1 # DO NOT CHANGE


@dataclass
class OpenO3VideoDatasetConfig:
    min_pixels: int = field(
        default=100_352,
        metadata={"help": "Minimum pixel budget per frame in the final returned frame stack."},
    )
    max_pixels: int = field(
        default=100_352 * 4,
        metadata={"help": "Maximum pixel budget per frame in the final returned frame stack."},
    )
    max_frames: int = field(
        default=32,
        metadata={"help": "Number of frames to uniformly sample to obtain the final annotated frame stack."},
    )
    sft_system_prompt_path: str = field(
        default="",
        metadata={"help": "Path to the instruction prompt template used for the system turn."},
    )


class OpenO3VideoDataset(Dataset):

    @classmethod
    def from_config(cls, config: "OpenO3VideoDatasetConfig") -> "OpenO3VideoDataset":
        return cls(
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_frames=config.max_frames,
        )

    def __init__(
        self,
        min_pixels: int = 100_352,
        max_pixels: int = 100_352 * 4,
        max_frames: int = 32,
    ) -> None:

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_frames = max_frames

        dataset, track_samples, cot_samples = self._load_dataset()
        self.dataset = dataset
        self.tracks = track_samples
        self.cots = cot_samples

    def __len__(self) -> int:
        return len(self.dataset)

    def _annotate_concepts(
        self,
        frames_np: np.ndarray,
        masks_dict: dict[str, np.ndarray],
        concepts: list[str],
        hw: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> np.ndarray:
        """Annotates a numpy array of frames with bounding boxes for each concept."""
        
        annotated = frames_np.copy()
        for concept in concepts:
            bboxes = masks_to_bboxes(masks_dict[concept], target_size=hw)
            for t, (x1, y1, x2, y2) in enumerate(bboxes.astype(int)):
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(annotated[t], (x1, y1), (x2, y2), color=tuple(color), thickness=thickness)
        return annotated

    def _load_video(self, sample: dict) -> tuple[np.ndarray, tuple[int, int]]:
        """Loads a video and returns the frames and its dimensions."""
        
        # load video
        video_dict = dict(
            video=get_video_path(sample),
            fps=FPS,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        frames = fetch_video(video_dict)
        frames_list = list(frames.permute(0, 2, 3, 1).cpu().numpy().astype("uint8"))  # list of (H,W,3) RGB

        # insert keyframes
        H, W = frames_list[0].shape[:2]
        splice_keyframes(sample, frames_list, H, W)
        
        return np.stack(frames_list), (H, W)

    def _load_dataset(self) -> tuple[list[dict], dict, dict]:

        # Video data
        dataset = load_dataset("json", data_files=GT_ANNOTATION_PATH, split="train")
        str_dataset = [s for s in dataset if s["task"] == "temporal-spatial free-form QA"]

        # SAM3 tracks
        run_meta_track, track_samples = load_track_manifest(SAM3_TRACKS_PATH)

        # CoT data
        cot_dir_path = os.path.join(COT_PATH, "refined_samples")
        cot_samples = {}
        json_files = glob(os.path.join(cot_dir_path, "*.json"))
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
                cot_samples[data['id']] = data

        # Filter out samples without masks and/or CoT
        def _has_masks_and_cot(sample: dict) -> bool:
            track = track_samples.get(sample["id"])
            cot = cot_samples.get(sample["id"])
            return track is not None and track.get("masks_file") is not None and cot is not None
        str_dataset = [s for s in str_dataset if _has_masks_and_cot(s)]

        return str_dataset, track_samples, cot_samples

    def __getitem__(self, idx: int) -> dict:
        """
        video loading/annotation process: 
            1. Load the video from the sample at FPS with the configured pixel budget
            2. Splice keyframes into the video 
            3. Annotate the frame stack (SAM3 tracks were generated using this spliced frame stack)
            3. Uniformly sample max_frames from this annotated frame stack (keyframe inclusion is hence not mandatory)
        """

        sample = self.dataset[idx]

        # load video (1 FPS)
        frames_np, hw = self._load_video(sample)

        # load COT
        cot = self.cots[sample["id"]]

        # load annotation tracks (1 FPS)
        masks_dict = load_track_masks(self.tracks[sample["id"]], SAM3_TRACKS_PATH)

        # get concepts
        concepts = list(masks_dict.keys())
        
        # annotate video
        color, thickness = BBOX_ANNOTATION_CONFIG["color"], BBOX_ANNOTATION_CONFIG["thickness"]
        annotated_np = self._annotate_concepts(frames_np, masks_dict, concepts, hw, color, thickness)
        
        # uniformly sample frames
        if len(annotated_np) > self.max_frames:
            indices = np.linspace(0, len(annotated_np) - 1, self.max_frames, dtype=int)
            annotated_np = annotated_np[indices]

        return {
            'annotated_frames': annotated_np,
            'question': sample['question'],
            'gt_cot': cot,
            'gt_answer': sample['answer'],
        }