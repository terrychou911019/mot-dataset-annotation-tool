import os
import shutil
import numpy as np

def split_tracklet(arr, tracklets_root, tid, split_frame):
    """
    Split a tracklet into two:
    - arr: numpy array of shape (num_tracklets+1, num_frames+1, 4), 1-based indexing
    - tracklets_root: root folder containing tracklet_xxxx directories
    - tid: tracklet id to split
    - split_frame: frame number (1-based) where split begins (inclusive)
    
    Returns:
      arr_new: updated numpy array with one extra tracklet
      new_tid: id of the newly created tracklet
    """

    num_tracklets, num_frames, _ = arr.shape
    new_tid = num_tracklets  # because arr is 1-based, last index = num_tracklets-1

    # 1. Expand numpy array to add a new tracklet row
    arr_new = np.full((num_tracklets + 1, num_frames, 4), np.nan, dtype=np.float32)
    arr_new[:num_tracklets, :, :] = arr

    # 2. Move data from split_frame onward to new_tid
    arr_new[new_tid, split_frame:, :] = arr_new[tid, split_frame:, :]
    arr_new[tid, split_frame:, :] = np.nan

    # 3. Update crops on disk
    old_dir = os.path.join(tracklets_root, f"tracklet_{tid:04d}")
    new_dir = os.path.join(tracklets_root, f"tracklet_{new_tid:04d}")
    os.makedirs(new_dir, exist_ok=True)

    # Move crops belonging to split_frame..end into the new_dir
    for fname in os.listdir(old_dir):
        if not fname.endswith(".jpg"):
            continue
        frame_id = int(os.path.splitext(fname)[0])
        if frame_id >= split_frame:
            shutil.move(os.path.join(old_dir, fname),
                        os.path.join(new_dir, fname))

    # Move manifest entries
    old_manifest = os.path.join(old_dir, "_manifest.csv")
    new_manifest = os.path.join(new_dir, "_manifest.csv")
    if os.path.exists(old_manifest):
        with open(old_manifest, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header = lines[0]
        with open(old_manifest, "w", encoding="utf-8") as f_old, \
             open(new_manifest, "w", encoding="utf-8") as f_new:
            f_old.write(header)
            f_new.write(header)
            for line in lines[1:]:
                frame_id = int(line.split(",")[0])
                if frame_id < split_frame:
                    f_old.write(line)
                else:
                    f_new.write(line)

    return arr_new, new_tid

# ===== Example usage =====
if __name__ == "__main__":
    seq_name = "seq01"
    arr = np.load(f"tracklets_array/{seq_name}.npy")  # Load existing array
    tracklets_root = f"tracklets_vis/{seq_name}"  # Root folder for tracklets
    tid = 18
    split_frame = 373

    arr_new, new_tid = split_tracklet(arr, tracklets_root, tid, split_frame)

    np.save(f"tracklets_array/{seq_name}.npy", arr_new)  # Save updated array

    print("New array shape:", arr_new.shape)
    print("New tracklet ID:", new_tid)