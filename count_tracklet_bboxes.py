import argparse
from pathlib import Path
import pandas as pd


def count_tracklet_bboxes(file_path: str, tracklet_id: int) -> int:
    """
    Count how many bounding boxes (bboxes) a given tracklet ID has.

    Args:
        file_path (str): Path to the annotation file.
        tracklet_id (int): Tracklet ID to count.

    Returns:
        int: Number of bounding boxes (rows) for the given tracklet ID.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path, header=None)
    return df[df[1] == tracklet_id].shape[0]


def main():
    parser = argparse.ArgumentParser(description="Count bboxes of a specific tracklet ID.")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("tracklet_id", type=int, help="Tracklet ID to count.")
    args = parser.parse_args()

    file_path = Path("gta_tracklets") / f"{args.sequence}.txt"
    count = count_tracklet_bboxes(file_path, args.tracklet_id)
    print(f"Tracklet {args.tracklet_id} has {count} bboxes in {args.sequence}.")


if __name__ == "__main__":
    main()
