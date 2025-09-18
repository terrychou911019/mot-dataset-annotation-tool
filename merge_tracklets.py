from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import numpy as np


def merge_tracklets(arr: np.ndarray, tid_a: int, tid_b: int, tracklets_root: str) -> tuple[np.ndarray, int]:
    """
    Merge two tracklets into a new one.

    Args:
        arr (np.ndarray): Tracklet array of shape (num_tracks+1, num_frames+1, 4).
        tid_a (int): First tracklet ID to merge (1-based).
        tid_b (int): Second tracklet ID to merge (1-based).
        tracklets_root (str): Root folder containing per-tracklet folders named as tracklet_XXXX.

    Returns:
        arr_new (np.ndarray): Updated array with one additional tracklet row after merging.
        new_tid (int): Newly created merged tracklet ID.
    """
    num_tracks, num_frames, _ = arr.shape

    if tid_a <= 0 or tid_b <= 0 or tid_a >= num_tracks or tid_b >= num_tracks:
        raise ValueError("Tracklet IDs out of range for 1-based array.")
    if tid_a == tid_b:
        raise ValueError("tid_a and tid_b must be different.")

    # Validate no overlap on the same frame
    a_has = ~np.isnan(arr[tid_a, :, 0])
    b_has = ~np.isnan(arr[tid_b, :, 0])
    overlap = np.where(a_has & b_has)[0]
    overlap = overlap[overlap >= 1]  # ignore padding index 0
    if overlap.size > 0:
        offending = ", ".join(str(int(f)) for f in overlap[:10])
        raise ValueError(
            f"Cannot merge: tracklets {tid_a} and {tid_b} overlap on frames {offending}..."
        )

    # Create new tracklet ID
    new_tid = num_tracks

    # Expand array
    arr_new = np.full((num_tracks + 1, num_frames, 4), np.nan, dtype=np.float32)
    arr_new[:num_tracks, :, :] = arr

    # Fill new tracklet row with union of tid_a and tid_b
    a_mask = ~np.isnan(arr[tid_a, :, 0])
    arr_new[new_tid, a_mask, :] = arr[tid_a, a_mask, :]
    b_mask = ~np.isnan(arr[tid_b, :, 0])
    arr_new[new_tid, b_mask, :] = arr[tid_b, b_mask, :]

    # Clear originals
    arr_new[tid_a, :, :] = np.nan
    arr_new[tid_b, :, :] = np.nan

    # Move image crops to new tracklet folder
    root = Path(tracklets_root)
    old_dir_a = root / f"tracklet_{tid_a:04d}"
    old_dir_b = root / f"tracklet_{tid_b:04d}"
    new_dir = root / f"tracklet_{new_tid:04d}"
    new_dir.mkdir(parents=True, exist_ok=True)

    def move_frame_images(src_dir: Path, dst_dir: Path) -> None:
        if not src_dir.exists():
            return
        for p in src_dir.iterdir():
            if p.suffix.lower() != ".jpg":
                continue
            if len(p.stem) != 6 or not p.stem.isdigit():
                continue
            dst = dst_dir / p.name
            if not dst.exists():
                shutil.move(str(p), str(dst))

    move_frame_images(old_dir_a, new_dir)
    move_frame_images(old_dir_b, new_dir)

    return arr_new, new_tid


def main():
    parser = argparse.ArgumentParser(description="Merge two tracklets into a new one.")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("tracklet_id_a", type=int, help="First tracklet ID to merge (1-based).")
    parser.add_argument("tracklet_id_b", type=int, help="Second tracklet ID to merge (1-based).")
    parser.add_argument("--arr_dir", type=str, default="tracklets_array",
                        help="Folder containing {sequence}.npy (default: tracklets_array).")
    parser.add_argument("--output_dir", type=str, default="tracklets_vis",
                        help="Root folder containing per-sequence tracklets (default: tracklets_vis).")
    args = parser.parse_args()

    arr_path = Path(args.arr_dir) / f"{args.sequence}.npy"
    arr = np.load(arr_path)
    tracklets_root = Path(args.output_dir) / args.sequence

    arr_new, new_tid = merge_tracklets(arr, args.tracklet_id_a, args.tracklet_id_b, str(tracklets_root))

    # overwrite the same npy
    np.save(arr_path, arr_new)

    print(f"New array shape: {arr_new.shape}")
    print(f"New merged tracklet ID: {new_tid:04d}")


if __name__ == "__main__":
    # Example: python merge_tracklets.py seq01 18 2   -> error
    # Example: python merge_tracklets.py seq01 18 20  -> success
    main()
