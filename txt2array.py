import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional


def txt2array(txt_path: str, num_frames: Optional[int] = None, num_tracks: Optional[int] = None) -> np.ndarray:
    """
    Load a MOT-style tracklet txt into a dense 3D numpy array.

    Array shape:
        (num_tracks + 1, num_frames + 1, 4)
    Indexing convention:
        - track IDs and frame IDs are assumed to be 1-based in the txt.
        - index 0 along track/frame is kept as padding (NaNs).
    Content:
        arr[track_id, frame, :] = [x, y, w, h] (float32), NaN if missing.

    Args:
        txt_path (str): Path to the annotation file.
        num_frames (int | None): Manually set the maximum frame index. If None, use max from file.
        num_tracks (int | None): Manually set the maximum track_id. If None, use max from file.

    Returns:
        np.ndarray: Dense array of shape (num_tracks+1, num_frames+1, 4), dtype=float32.
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Define/parse columns (MOT format compatible)
    cols = ["frame", "track_id", "x", "y", "w", "h", "score", "i1", "i2", "i3"]
    df = pd.read_csv(path, header=None, names=cols, usecols=[0, 1, 2, 3, 4, 5])

    # Ensure dtypes
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df[["x", "y", "w", "h"]] = df[["x", "y", "w", "h"]].astype(np.float32)

    # Determine sizes if not provided
    max_frame = int(df["frame"].max()) if not df.empty else 0
    max_track = int(df["track_id"].max()) if not df.empty else 0
    if num_frames is None:
        num_frames = max_frame
    if num_tracks is None:
        num_tracks = max_track

    # Initialize with NaNs
    arr = np.full((num_tracks + 1, num_frames + 1, 4), np.nan, dtype=np.float32)

    if not df.empty:
        frames = df["frame"].to_numpy(dtype=int)
        tracks = df["track_id"].to_numpy(dtype=int)
        boxes = df[["x", "y", "w", "h"]].to_numpy(dtype=np.float32)

        # Vectorized assignment
        arr[tracks, frames] = boxes

    return arr


def main():
    parser = argparse.ArgumentParser(description="Convert MOT-style txt to dense numpy array.")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("--txt_dir", type=str, default="gta_tracklets",
                        help="Folder that contains {sequence}.txt (default: gta_tracklets)")
    parser.add_argument("--output_dir", type=str, default="tracklets_array",
                        help="Folder to save {sequence}.npy (default: tracklets_array)")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Override max frame index; if omitted, inferred from file")
    parser.add_argument("--num_tracks", type=int, default=None,
                        help="Override max track id; if omitted, inferred from file")
    args = parser.parse_args()

    txt_path = Path(args.txt_dir) / f"{args.sequence}.txt"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.sequence}.npy"

    arr = txt2array(str(txt_path), num_frames=args.num_frames, num_tracks=args.num_tracks)
    np.save(output_path, arr)

    print(f"Saved array to: {output_path}")
    print(f"shape: {arr.shape}  (tracks+1, frames+1, 4)")
    print(f"dtype: {arr.dtype}")


if __name__ == "__main__":
    main()
