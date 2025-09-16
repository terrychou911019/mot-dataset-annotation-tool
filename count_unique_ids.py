import argparse
import pandas as pd
from pathlib import Path


def count_unique_ids(seq_name: str, root: str = "gta_tracklets") -> int:
    """
    Count the number of unique IDs in a given sequence annotation file.

    Args:
        seq_name (str): Name of the sequence (e.g., 'seq01').
        root (str): Root directory where annotation files are stored.

    Returns:
        int: Number of unique IDs.
    """
    file_path = Path(root) / f"{seq_name}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, header=None)
    return df[1].nunique()


def main():
    parser = argparse.ArgumentParser(description="Count unique IDs in a sequence annotation file.")
    parser.add_argument(
        "sequence",
        type=str,
        help="Sequence name (e.g., seq01). The file is expected at gta_tracklets/{sequence}.txt",
    )
    args = parser.parse_args()

    unique_ids = count_unique_ids(args.sequence)
    print(f"Unique IDs in {args.sequence}: {unique_ids}")


if __name__ == "__main__":
    main()
