import argparse
from pathlib import Path
import pandas as pd


def count_unique_ids(file_path: str) -> int:
    """
    Count the number of unique IDs in a given annotation file.

    Args:
        file_path (str): Path to the annotation file. 

    Returns:
        int: Number of unique IDs.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path, header=None)
    return df[1].nunique()


def main():
    parser = argparse.ArgumentParser(description="Count unique IDs in a sequence annotation file.")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    args = parser.parse_args()

    file_path = Path("gta_tracklets") / f"{args.sequence}.txt"
    unique_ids = count_unique_ids(file_path)
    print(f"Unique IDs in {args.sequence}: {unique_ids}")


if __name__ == "__main__":
    main()
