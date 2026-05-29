import os 
DATA_ROOT = "/projects/torresani-lab/ajay/datasets/Open-o3-Video/"

def get_video_path(example: dict[str, any], data_root: str = DATA_ROOT) -> str:
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

def get_key_frame_path(key_frame: dict[str, any], example: dict[str, any], data_root: str = DATA_ROOT) -> str:
    root = os.path.join(data_root, "videos")

    if example.get("source") == "STR_plm_rdcap":
        key_frame_root = os.path.join(root, "stgr/plm/kfs")
    else:
        key_frame_root = os.path.join(root, "stgr/temporal_grounding/kfs")

    path = os.path.join(key_frame_root, key_frame["path"])

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    return path