import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_array(arr, seq_name, save_path=None):
    """
    Visualize tracklet existence as a Gantt-like chart (1-based indexing).
    X-axis: frame id (1-based, integer ticks)
    Y-axis: tracklet id (1-based, all tracklet ids shown)
    A horizontal line is drawn when a tracklet has consecutive bboxes.
    """
    num_tracks, num_frames, _ = arr.shape

    fig, ax = plt.subplots(figsize=(12, 6))

    for tid in range(1, num_tracks):  # skip index 0, since 1-based
        # Boolean mask of which frames have bbox
        has_box = ~np.isnan(arr[tid, :, 0])

        start = None
        for f in range(1, num_frames):  # skip index 0
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

    # Show integer frame ids (1-based)
    ax.set_xticks(np.arange(1, num_frames + 1, 500))
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ===== Example usage =====
if __name__ == "__main__":
    seq_name = "seq01"
    stage = "split"
    arr = np.load(f"tracklets_array/{seq_name}.npy")  #

    visualize_array(arr, seq_name, f"demo/{seq_name}/{stage}_stage.png")
