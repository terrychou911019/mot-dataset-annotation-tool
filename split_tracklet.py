from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import numpy as np


def split_tracklet(arr: np.ndarray, tid: int, split_frame: int, tracklets_root: str) -> tuple[np.ndarray, int]:
    """
    Split a tracklet into two at a given frame (inclusive).

    Args:
        arr (np.ndarray): Tracklet array of shape (num_tracks+1, num_frames+1, 4).
        tid (int): Tracklet ID to split (1-based).
        split_frame (int): Frame index (1-based) where the split begins (inclusive).
        tracklets_root (str): Root folder containing per-tracklet folders named as tracklet_XXXX.  

    Returns:
        tuple[np.ndarray, int]: (arr_new, new_tid)
            - arr_new: Updated array with one additional tracklet row (1-based indexing preserved).
            - new_tid: Newly created tracklet ID.
    """
    num_tracks, num_frames, _ = arr.shape  # already includes +1 for padding

    if not (1 <= tid < num_tracks):
        raise ValueError(f"tid out of range: {tid} (valid: 1..{num_tracks-1})")
    if not (1 <= split_frame < num_frames):
        raise ValueError(f"split_frame out of range: {split_frame} (valid: 1..{num_frames-1})")

    # The new tracklet ID is the last index
    new_tid = num_tracks

    # Expand the array by adding one more tracklet row
    arr_new = np.full((num_tracks + 1, num_frames, 4), np.nan, dtype=np.float32)
    arr_new[:num_tracks, :, :] = arr

    # Move detections from split_frame onward to the new tracklet
    arr_new[new_tid, split_frame:, :] = arr_new[tid, split_frame:, :]
    arr_new[tid, split_frame:, :] = np.nan

    # Update crops and manifests on disk
    root = Path(tracklets_root)
    old_dir = root / f"tracklet_{tid:04d}"
    new_dir = root / f"tracklet_{new_tid:04d}"
    new_dir.mkdir(parents=True, exist_ok=True)

    if old_dir.exists():
        # Move images with frame_id >= split_frame
        for p in old_dir.iterdir():
            if p.suffix.lower() != ".jpg":
                continue
            try:
                frame_id = int(p.stem)
            except ValueError:
                continue
            if frame_id >= split_frame:
                shutil.move(str(p), str(new_dir / p.name))

        # Update manifest files
        old_manifest = old_dir / "_manifest.csv"
        new_manifest = new_dir / "_manifest.csv"
        if old_manifest.exists():
            lines = old_manifest.read_text(encoding="utf-8").splitlines()
            if lines:
                header = lines[0]
                with old_manifest.open("w", encoding="utf-8", newline="") as f_old, \
                     new_manifest.open("w", encoding="utf-8", newline="") as f_new:
                    f_old.write(header + "\n")
                    f_new.write(header + "\n")
                    for line in lines[1:]:
                        if not line.strip():
                            continue
                        frame_id = int(line.split(",")[0])
                        if frame_id < split_frame:
                            f_old.write(line + "\n")
                        else:
                            f_new.write(line + "\n")

    return arr_new, new_tid


def main():
    parser = argparse.ArgumentParser(description="Split a tracklet into two starting at a given frame (inclusive).")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("tracklet_id", type=int, help="Tracklet ID to split (1-based).")
    parser.add_argument("split_frame", type=int, help="Frame index where split begins (1-based).")
    parser.add_argument("--arr_dir", type=str, default="tracklets_array",
                        help="Folder containing {sequence}.npy (default: tracklets_array).")
    parser.add_argument("--output_dir", type=str, default="tracklets_vis",
                        help="Root folder containing per-sequence tracklets (default: tracklets_vis).")
    args = parser.parse_args()

    arr_path = Path(args.arr_dir) / f"{args.sequence}.npy"
    arr = np.load(arr_path)
    tracklets_root = Path(args.output_dir) / args.sequence

    arr_new, new_tid = split_tracklet(arr, args.tracklet_id, args.split_frame, str(tracklets_root))

    # overwrite the same npy
    np.save(arr_path, arr_new)

    print(f"New array shape: {arr_new.shape}")
    print(f"New tracklet ID: {new_tid:04d}")


if __name__ == "__main__":
    # python split_tracklet.py seq01 18 373
    main()
