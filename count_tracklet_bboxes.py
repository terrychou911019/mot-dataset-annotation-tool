# count_bboxes.py
import argparse
import pandas as pd
from pathlib import Path


def count_tracklet_bboxes(seq_name: str, tracklet_id: int, root: str = "gta_tracklets") -> int:
    """
    Count how many bounding boxes (bboxes) a given tracklet ID has.

    Args:
        seq_name (str): Sequence name (e.g., 'seq01').
        tracklet_id (int): Tracklet ID to count.
        root (str): Root directory where annotation files are stored.

    Returns:
        int: Number of bounding boxes (rows) for the given tracklet ID.
    """
    file_path = Path(root) / f"{seq_name}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, header=None)
    return df[df[1] == tracklet_id].shape[0]


def main():
    parser = argparse.ArgumentParser(description="Count bboxes of a specific tracklet ID.")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("tracklet_id", type=int, help="Tracklet ID to count.")
    args = parser.parse_args()

    count = count_tracklet_bboxes(args.sequence, args.tracklet_id)
    print(f"Tracklet {args.tracklet_id} has {count} bboxes in {args.sequence}.")


if __name__ == "__main__":
    main()
