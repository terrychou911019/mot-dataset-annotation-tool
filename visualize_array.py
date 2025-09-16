from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Optional


def visualize_array(arr: np.ndarray, seq_name: str, save_path: Optional[str] = None) -> None:
    """
    Visualize tracklet existence as a Gantt-like chart (1-based indexing).

    Args:
        arr (np.ndarray): Tracklet array of shape (num_tracks+1, num_frames+1, 4).
        seq_name (str): Sequence name (for title).
        save_path (str | None): Optional path to save the figure. If None, only display.
    """
    num_tracks, num_frames, _ = arr.shape
    fig, ax = plt.subplots(figsize=(12, 6))

    for tid in range(1, num_tracks):  # skip index 0 (padding)
        has_box = ~np.isnan(arr[tid, :, 0])
        start = None
        for f in range(1, num_frames):  # skip index 0 (padding)
            if has_box[f] and start is None:
                start = f
            elif (not has_box[f] or f == num_frames - 1) and start is not None:
                end = f if has_box[f] else f - 1
                ax.hlines(tid, start, end, colors="blue", linewidth=2)
                start = None

    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Tracklet ID")
    ax.set_title(f"{seq_name} Tracklet Existence (Gantt-like View)")

    # Show all tracklet ids (1-based)
    ax.set_yticks(np.arange(1, num_tracks))
    ax.set_ylim(0.5, num_tracks - 0.5)

    # Show integer frame ids (1-based), step = 500
    ax.set_xticks(np.arange(1, num_frames + 1, 500))

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize tracklet existence as Gantt-like chart.")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("stage", type=str, help="Stage name (gta, split, merge, interpolate).")
    parser.add_argument("--arr_dir", type=str, default="tracklets_array",
                        help="Folder containing {sequence}.npy (default: tracklets_array).")
    parser.add_argument("--output_dir", type=str, default="demo",
                        help="Folder to save visualization (default: demo).")
    args = parser.parse_args()

    arr_path = Path(args.arr_dir) / f"{args.sequence}.npy"
    arr = np.load(arr_path)
    output_path = Path(args.output_dir) / args.sequence / f"{args.stage}_stage.png"
    
    visualize_array(arr, args.sequence, str(output_path))


if __name__ == "__main__":
    main()
