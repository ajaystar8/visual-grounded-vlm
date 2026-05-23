import os
import numpy as np
import cv2


def _to_numpy_frames(frames):
    """Convert frames (tensor, list of tensors, or array) to (T, H, W, 3) uint8 numpy array."""
    import torch
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    elif isinstance(frames, (list, tuple)) and len(frames) > 0 and isinstance(frames[0], torch.Tensor):
        frames = np.stack([f.detach().cpu().numpy() for f in frames])
    else:
        frames = np.array(frames)
    if frames.ndim == 4 and frames.shape[1] == 3:  # (T, 3, H, W) -> (T, H, W, 3)
        frames = frames.transpose(0, 2, 3, 1)
    elif frames.ndim == 4 and frames.shape[1] == 1:  # (T, 1, H, W) -> (T, H, W, 3) grayscale
        frames = np.repeat(frames.transpose(0, 2, 3, 1), 3, axis=-1)
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        else:
            frames = frames.clip(0, 255).astype(np.uint8)
    return frames


def show_video(frames=None, video_path=None, fps=5, width=480, save_path=None, masks=None, mask_color=(255, 0, 0), mask_alpha=0.5):
    """Save frames as H.264 mp4 and/or display inline in notebook.

    Args:
        frames: (T, C, H, W) or (T, H, W, C) tensor/array, or list of frames.
        masks: optional (T, 1, H, W) or (T, H, W) binary mask tensor/array to overlay.
        mask_color: RGB color for the mask overlay (default red).
        mask_alpha: transparency of the mask overlay (0=invisible, 1=opaque).
    """
    import subprocess
    import base64
    from IPython.display import HTML, display

    if video_path and frames is None:
        path = save_path or "/tmp/_preview.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vcodec", "libx264", path], check=True, capture_output=True)
    elif frames is not None:
        path = save_path or "/tmp/_preview.mp4"
        frames = _to_numpy_frames(frames)

        if masks is not None:
            masks = _to_numpy_frames(masks)  # -> (T, H, W, 3)
            # convert to binary (T, H, W) bool
            mask_bin = masks[..., 0] > 128
            overlay_color = np.array(mask_color, dtype=np.uint8)
            blended = frames.copy().astype(np.float32)
            for t in range(len(frames)):
                m = mask_bin[t]
                blended[t][m] = (1 - mask_alpha) * frames[t][m] + mask_alpha * overlay_color
            frames = blended.clip(0, 255).astype(np.uint8)

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
        writer.release()

        tmp = path + ".tmp.mp4"
        os.rename(path, tmp)
        subprocess.run(["ffmpeg", "-y", "-i", tmp, "-vcodec", "libx264", path], check=True, capture_output=True)
        os.remove(tmp)
    else:
        raise ValueError("Provide either frames or video_path")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    display(HTML(f"""
    <video width="{width}" controls>
      <source src="data:video/mp4;base64,{b64}" type="video/mp4">
    </video>
    """))
    return path
