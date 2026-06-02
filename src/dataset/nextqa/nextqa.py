import pandas as pd
import os.path as osp
import json

from qwen_vl_utils import fetch_video
from torch.utils.data import Dataset

DATASET_ROOT = "/projects/torresani-lab/ajay/datasets/nextqa"
VIDEO_CONFIG = {
    "total_pixels": 2048 * 28 * 28,
    "min_pixels": 256 * 28 * 28,
    "nframes": 32,
}

class NextQADataset(Dataset):

    VIDEO_DIR = osp.join(DATASET_ROOT, "NExTVideo")
    MAP_PATH = osp.join(DATASET_ROOT, "map_vid_vidorID.json")

    def __init__(
        self,
        split: str,
        load_video: bool=True,
        video_config: dict = VIDEO_CONFIG,
    ):  
        # Load video ID to path mapping
        self.vid2path = self._get_vid_to_path_map()

        # GT annotations
        self.gt_annotations_df = self._get_gt_annotations(split)

        self.load_video = load_video
        self.video_config = video_config
    
    def _get_vid_to_path_map(self):
        if not osp.exists(self.MAP_PATH):
            raise FileNotFoundError(f"Mapping file not found at {self.MAP_PATH}")
        with open(self.MAP_PATH, 'r') as f:
            return json.load(f)
        
    def _get_video_path(self, video_id: str):
        vid_path = osp.join(self.VIDEO_DIR, self.vid2path[str(video_id)] + ".mp4")
        if not osp.exists(vid_path):
            raise FileNotFoundError(f"Video file not found at {vid_path}")
        return vid_path
    
    def _get_gt_annotations(self, split: str):
        assert split in ["train", "val", "test"], f"Invalid split: {split}. Must be one of ['train', 'val', 'test']."
        annotation_path = osp.join(DATASET_ROOT, "annotations_nextqa", f"{split}.csv")

        if not osp.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found at {annotation_path}")
        return pd.read_csv(annotation_path)
    
    def _load_video_frames(self, video_path: str):
        video_config = self.video_config.copy()
        video_config["video"] = video_path
        return fetch_video(video_config)
    
    def __getitem__(self, idx):
        row = self.gt_annotations_df.iloc[idx]
        video_id = row["video"]
        question = row["question"]

        # Format options string for multiple choice questions
        options = ['A', 'B', 'C', 'D', 'E']
        formatted_options = [
            f"{letter}. {row[f'a{i}'].strip()}"
            for i, letter in enumerate(options)
        ]
        options_str = "\n".join(formatted_options)
        answer = formatted_options[int(row["answer"])]

        video_path = self._get_video_path(video_id)
        video_frames = None
        if self.load_video:
            video_frames = self._load_video_frames(video_path)

        return {
            "question": question,
            "answer": answer, # e.g. "(A) option text"
            "answer_option": options[int(row["answer"])], # e.g. simply "A", "B", etc.
            "vid": str(row['video']),
            "qid": str(row["qid"]),
            "type": str(row["type"]),
            "options_str": options_str,
            "video_path": video_path,
            "video_frames": video_frames,
        }

    def __len__(self):
        return len(self.gt_annotations_df)

    def __repr__(self):
        n_videos = self.gt_annotations_df["video"].nunique()
        n_qa = len(self.gt_annotations_df)
        return f"{type(self).__name__}(videos={n_videos}, qa_pairs={n_qa})"