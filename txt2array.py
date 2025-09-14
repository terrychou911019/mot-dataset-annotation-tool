import os
import numpy as np
import pandas as pd

def txt2array(file_path, num_frames=None, num_tracks=None):
    """
    Load MOT-style tracklet file into a 3D numpy array.
    Shape = (num_tracks+1, num_frames+1, 4)
    Each element stores a bounding box (x, y, w, h).
    Missing detections are filled with np.nan.
    """
    # Define column names for MOT format
    cols = ["frame", "track_id", "x", "y", "w", "h", "score", "i1", "i2", "i3"]
    df = pd.read_csv(file_path, header=None, names=cols)

    # Convert frame_id and track_id to integer
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    # If not specified, use maximum values from data
    if num_frames is None:
        num_frames = df["frame"].max()
    if num_tracks is None:
        num_tracks = df["track_id"].max()

    # Initialize array: track_id × frame × bbox(4)
    # Fill with NaN to represent missing detections
    arr = np.full((num_tracks + 1, num_frames + 1, 4), np.nan, dtype=np.float32)

    # Insert detections into the array
    for _, row in df.iterrows():
        f = int(row["frame"])
        t = int(row["track_id"])
        arr[t, f, :] = [row["x"], row["y"], row["w"], row["h"]]

    return arr

# ===== Example usage =====
if __name__ == "__main__":
    seq_name = "seq01"
    file_path = f"gta_tracklets/{seq_name}.txt"  # Path to your tracklet file
    arr = txt2array(file_path)
    
    os.makedirs(f"tracklets_array", exist_ok=True)
    np.save(f"tracklets_array/{seq_name}.npy", arr)  # Save array to .npy file

    print("Array shape:", arr.shape)  # (num_tracks+1, num_frames+1, 4)
