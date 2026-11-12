"""Video recorder utility for Super Mario MO-PPO.

Exports gameplay episodes as MP4 videos with an informative HUD overlay
showing position, remaining time, coins, score, and completion status.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

logger = logging.getLogger(__name__)


def _draw_hud(
    frame: np.ndarray,
    step_idx: int,
    info: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Draw a HUD banner on the frame displaying Mario's stats."""
    if not HAS_CV2:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw semi-transparent header bar (top 32 pixels)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Info text formatting
    x_pos = info.get("x_pos", 0) if info else 0
    game_time = info.get("time", 0) if info else 0
    coins = info.get("coins", 0) if info else 0
    score = info.get("score", 0) if info else 0
    flag_get = info.get("flag_get", False) if info else False

    status = "CLEAR!" if flag_get else "RUNNING"
    status_color = (0, 255, 0) if flag_get else (255, 255, 255)

    line1 = f"STEP: {step_idx:04d} | X: {x_pos:04d} | TIME: {game_time:03d} | COIN: {coins:02d} | PTS: {score:06d}"
    
    cv2.putText(
        annotated,
        line1,
        (6, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"STATUS: {status}",
        (6, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        status_color,
        1,
        cv2.LINE_AA,
    )

    return annotated


def save_episode_video(
    frames: Union[List[np.ndarray], np.ndarray],
    filepath: str,
    fps: int = 30,
    hud_info: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
) -> str:
    """Save an episode of RGB frames to an MP4 video file.

    Args:
        frames: Sequence of (H, W, 3) images (RGB).
        filepath: Destination path ending in .mp4.
        fps: Frames per second (default: 30).
        hud_info: Optional per-step dictionary or list of dictionaries containing info
                  with keys ('x_pos', 'time', 'coins', 'score', 'flag_get').

    Returns:
        Absolute path to the created video file.
    """
    if len(frames) == 0:
        logger.warning("Empty frames list received; skipping video creation.")
        return ""

    filepath = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    first_frame = frames[0]
    height, width = first_frame.shape[:2]

    processed_frames: List[np.ndarray] = []
    for idx, frame in enumerate(frames):
        curr_frame = frame.copy()
        # If frame is grayscale (H, W) or (1, H, W), convert to 3 channels
        if curr_frame.ndim == 2:
            curr_frame = np.stack([curr_frame] * 3, axis=-1)
        elif curr_frame.ndim == 3 and curr_frame.shape[0] in (1, 4):
            # Transpose if channel-first (C, H, W)
            if curr_frame.shape[0] == 4:
                # Stood from frame stack: use the latest frame
                curr_frame = curr_frame[-1]
                curr_frame = np.stack([curr_frame] * 3, axis=-1)
            elif curr_frame.shape[0] == 1:
                curr_frame = np.squeeze(curr_frame, axis=0)
                curr_frame = np.stack([curr_frame] * 3, axis=-1)

        # Get HUD info for this frame
        info_for_step = None
        if isinstance(hud_info, list) and idx < len(hud_info):
            info_for_step = hud_info[idx]
        elif isinstance(hud_info, dict):
            info_for_step = hud_info

        frame_with_hud = _draw_hud(curr_frame, idx, info_for_step)
        processed_frames.append(frame_with_hud)

    # Attempt OpenCV VideoWriter first
    if HAS_CV2:
        try:
            # OpenCV expects BGR
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(filepath, fourcc, float(fps), (width, height))
            for f in processed_frames:
                bgr_f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.shape[-1] == 3 else f
                out.write(bgr_f)
            out.release()
            logger.info("Successfully saved video using OpenCV: %s (%d frames)", filepath, len(frames))
            return filepath
        except Exception as cv_err:
            logger.warning("OpenCV VideoWriter failed: %s. Trying imageio fallback.", cv_err)

    # Fallback to imageio
    if HAS_IMAGEIO:
        try:
            with imageio.get_writer(filepath, fps=fps, codec="libx264") as writer:
                for f in processed_frames:
                    writer.append_data(f)
            logger.info("Successfully saved video using imageio: %s", filepath)
            return filepath
        except Exception as io_err:
            logger.error("imageio failed to write video: %s", io_err)

    logger.error("Failed to write video: Neither OpenCV nor imageio succeeded.")
    return ""
