from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def array2txt(arr: np.ndarray, txt_path: str) -> None:
    """
    Convert a dense tracklet array back into MOT-style txt format.

    Input array:
        Shape: (num_tracks+1, num_frames+1, 4), dtype=float32
        Indexing: 1-based (0 is padding along track and frame)
        Content: arr[tid, frame] = [x, y, w, h] or NaN if missing

    Output format (one row per bbox):
        frame, track_id, x, y, w, h, 1, 1, 1

    Args:
        arr (np.ndarray): Dense tracklet array (num_tracks+1, num_frames+1, 4).
        txt_path (str): Path to the output txt file.
    """
    num_tracks, num_frames, _ = arr.shape
    path = Path(txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        # Iterate over tracks and frames (skip index 0 because of padding)
        for frame in range(1, num_frames):
            for tid in range(1, num_tracks):
                bbox = arr[tid, frame]
                if np.any(np.isnan(bbox)):
                    continue
                x, y, w, h = bbox.tolist()
                f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")


def main():
    parser = argparse.ArgumentParser(description="Convert dense numpy array to MOT-style txt.")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("--arr_dir", type=str, default="tracklets_array",
                        help="Folder containing {sequence}.npy (default: tracklets_array).")
    parser.add_argument("--output_dir", type=str, default="final_tracklets",
                        help="Folder to save {sequence}.txt (default: final_tracklets).")
    args = parser.parse_args()

    arr_path = Path(args.arr_dir) / f"{args.sequence}.npy"
    arr = np.load(arr_path)
    txt_path = Path(args.output_dir) / f"{args.sequence}.txt"

    array2txt(arr, str(txt_path))

    print(f"Saved txt to: {txt_path}")
    print(f"array shape: {arr.shape}  (tracks+1, frames+1, 4)")
    print(f"lines written: {sum(1 for _ in open(txt_path, 'r', encoding='utf-8'))}")


if __name__ == "__main__":
    main()
