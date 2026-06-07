import os
import json
import numpy as np

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


def splice_keyframes(sample: dict, frames_list: list, H: int, W: int) -> list[dict]:
    """Splice each unique concept's keyframe image into the 1fps frame list as a grounding seed.

    For each unique concept in the sample, the keyframe image (resized to H x W) is inserted
    into frames_list right after the 1fps frame nearest to the keyframe timestamp. This is the
    canonical way to build the augmented frame stack used for SAM3 tracking on this dataset.

    Args:
        sample: one STR dataset item (must have 'key_frames' and 'key_items').
        frames_list: list of (H, W, 3) uint8 RGB arrays sampled at 1fps — mutated in-place.
        H, W: frame height and width (keyframe images are resized to match).
    Returns:
        entities: list of dicts with keys:
            concept   — concept label string
            seed_idx  — index into the (now augmented) frames_list of the spliced keyframe
            gt_bbox   — pixel-space (x1, y1, x2, y2) GT bbox on the seed frame
    """
    import cv2

    n_sampled = len(frames_list)

    # collect one entry per unique concept, sorted by timestamp so splices go left-to-right
    seen, kfs = set(), []
    for kf in sample["key_frames"]:
        for concept, bboxes in sample["key_items"].get(str(kf["idx"]), {}).items():
            if concept in seen:
                continue
            seen.add(concept)
            kf_bgr = cv2.imread(get_key_frame_path(kf, sample))
            kf_img = cv2.resize(cv2.cvtColor(kf_bgr, cv2.COLOR_BGR2RGB), (W, H))
            gt = bboxes[0]
            kfs.append(dict(
                pos=min(int(round(kf["time"])), n_sampled - 1),
                concept=concept,
                img=kf_img,
                gt_bbox=(int(gt[0] * W), int(gt[1] * H), int(gt[2] * W), int(gt[3] * H)),
            ))
    kfs.sort(key=lambda k: k["pos"])

    # insert each keyframe after its nearest 1fps frame; offset corrects for prior insertions shifting indices
    entities = []
    for offset, k in enumerate(kfs):
        seed_idx = k["pos"] + 1 + offset
        frames_list.insert(seed_idx, k["img"])
        entities.append(dict(concept=k["concept"], seed_idx=seed_idx, gt_bbox=k["gt_bbox"]))

    return entities


def load_track_manifest(results_dir: str) -> tuple[dict, list[dict]]:
    """Load manifest.json from a sam3_track results dir; returns (run_meta, samples).

    Args:
        results_dir: path to the experiment output dir (e.g. .../openo3_str_sam3_track_test).
    Returns:
        run_meta: dict of run-level metadata (config, timing, device, etc.)
        samples: dict mapping sample id -> per-sample dict (concepts, frame_size, masks_file, etc.)
    """
    manifest = json.load(open(os.path.join(results_dir, "manifest.json")))
    raw = manifest["samples"]
    samples = raw if isinstance(raw, dict) else {s["id"]: s for s in raw}
    return manifest["run"], samples


def load_track_masks(sample_entry: dict, results_dir: str) -> dict[str, np.ndarray]:
    """Load the compressed masks npz for one sample entry from the manifest.

    Args:
        sample_entry: one element from the `samples` list returned by load_track_manifest.
        results_dir: same results dir passed to load_track_manifest.
    Returns:
        dict mapping concept name -> (T, H, W) bool array. Empty dict if no masks were saved.
    """
    if sample_entry["masks_file"] is None:
        return {}
    npz = np.load(os.path.join(results_dir, sample_entry["masks_file"]))
    return {concept: npz[concept].astype(bool) for concept in npz.files}


def masks_to_bboxes(masks: np.ndarray, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Derive per-frame bboxes from a (T, H, W) bool mask array.

    Args:
        masks: (T, H, W) bool array from load_track_masks.
        target_size: optional (H_out, W_out) to scale bboxes into pixel coords for that resolution.
                     If None, returns normalized [0, 1] xyxy coords.
    Returns:
        (T, 4) float32 array of xyxy bboxes. Rows are [0,0,0,0] for empty-mask frames.
    """
    T = len(masks)
    bboxes = np.zeros((T, 4), dtype=np.float32)
    for t, m in enumerate(masks):
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        H, W = m.shape
        bboxes[t] = [xs.min() / W, ys.min() / H, xs.max() / W, ys.max() / H]
    # scale to target pixel space if requested
    if target_size is not None:
        H_out, W_out = target_size
        bboxes *= [W_out, H_out, W_out, H_out]
    return bboxes