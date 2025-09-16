import os
import shutil
import numpy as np

def merge_tracklets(arr, tracklets_root, tid_a, tid_b):
    """
    Merge two tracklets into a new one:
    - arr: numpy array of shape (num_tracklets+1, num_frames+1, 4), 1-based indexing
    - tracklets_root: root folder containing tracklet_xxxx directories
    - tid_a, tid_b: tracklet ids to merge

    Rules:
      - Before merging, ensure the two tracklets NEVER appear on the same frame.
      - Create a new tracklet id (max_id + 1), copy all boxes from tid_a and tid_b into it.
      - Clear (set to NaN) the original rows of tid_a and tid_b to avoid duplication.
      - On disk, move/merge crops from both tracklets into the new tracklet folder and
        merge their manifests into a single _manifest.csv.

    Returns:
      arr_new: updated numpy array with one extra tracklet
      new_tid: id of the newly created merged tracklet
    """
    num_tracklets, num_frames, _ = arr.shape
    if tid_a <= 0 or tid_b <= 0 or tid_a >= num_tracklets or tid_b >= num_tracklets:
        raise ValueError("Tracklet ids out of range for 1-based array.")
    if tid_a == tid_b:
        raise ValueError("tid_a and tid_b must be different.")

    # ---- 1) Validate no overlap on the same frame ----
    # has box if x is not NaN (any of x,y,w,h works; use x=..., index 0)
    a_has = ~np.isnan(arr[tid_a, :, 0])
    b_has = ~np.isnan(arr[tid_b, :, 0])
    overlap = np.where(a_has & b_has)[0]  # 0..num_frames-1 (1-based indexing in content)
    # Ignore index 0 since it's the 1-based padding column
    overlap = overlap[overlap >= 1]
    if overlap.size > 0:
        # Show a few offending frames for easier debugging
        offending = ", ".join(str(int(f)) for f in overlap[:10])
        raise ValueError(
            f"Cannot merge: tracklets {tid_a} and {tid_b} overlap on frames: {offending}..."
        )

    # ---- 2) Expand numpy array to add a new tracklet row ----
    new_tid = num_tracklets  # because indices are 0..num_tracklets and 1-based content lives in 1..num_tracklets-1
    arr_new = np.full((num_tracklets + 1, num_frames, 4), np.nan, dtype=np.float32)
    arr_new[:num_tracklets, :, :] = arr

    # ---- 3) Compose new tracklet row = union of tid_a and tid_b (no conflicts guaranteed) ----
    # Start empty, then fill from A then B (order doesn't matter due to no overlap)
    # Copy frames where A has boxes
    a_mask = ~np.isnan(arr_new[tid_a, :, 0])
    arr_new[new_tid, a_mask, :] = arr_new[tid_a, a_mask, :]
    # Copy frames where B has boxes
    b_mask = ~np.isnan(arr_new[tid_b, :, 0])
    arr_new[new_tid, b_mask, :] = arr_new[tid_b, b_mask, :]

    # Clear originals to avoid duplication
    arr_new[tid_a, :, :] = np.nan
    arr_new[tid_b, :, :] = np.nan

    # ---- 4) Filesystem: merge per-frame crops and manifests ----
    old_dir_a = os.path.join(tracklets_root, f"tracklet_{tid_a:04d}")
    old_dir_b = os.path.join(tracklets_root, f"tracklet_{tid_b:04d}")
    new_dir   = os.path.join(tracklets_root, f"tracklet_{new_tid:04d}")
    os.makedirs(new_dir, exist_ok=True)

    # Helper to move all per-frame images (accept common extensions) into new_dir
    def move_frame_images(src_dir, dst_dir):
        if not os.path.isdir(src_dir):
            return
        for fname in os.listdir(src_dir):
            base, ext = os.path.splitext(fname)
            if len(base) != 6 or not base.isdigit():
                continue  # skip non frame-named files
            if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            # If both tracklets happen to have same frame file name (shouldn't happen due to no-overlap),
            # the earlier check prevents it. Still, guard with overwrite=False semantics.
            if not os.path.exists(dst):
                shutil.move(src, dst)

    move_frame_images(old_dir_a, new_dir)
    move_frame_images(old_dir_b, new_dir)

    # Merge manifests
    def read_manifest_lines(dir_path):
        path = os.path.join(dir_path, "_manifest.csv")
        if not os.path.exists(path):
            return None, []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header = lines[0] if lines else "frame,x,y,w,h,filename\n"
        content = lines[1:] if len(lines) > 1 else []
        return header, content

    hdr_a, cont_a = read_manifest_lines(old_dir_a)
    hdr_b, cont_b = read_manifest_lines(old_dir_b)
    # Prefer existing header; fallback to default
    header = hdr_a or hdr_b or "frame,x,y,w,h,filename\n"

    new_manifest = os.path.join(new_dir, "_manifest.csv")
    with open(new_manifest, "w", encoding="utf-8") as f_new:
        f_new.write(header)
        # Write all lines; since no overlap, we can just append
        for line in cont_a:
            f_new.write(line)
        for line in cont_b:
            f_new.write(line)

    # Rewrite old manifests to keep only header (acts like "emptied")
    def truncate_manifest(dir_path, header_line):
        path = os.path.join(dir_path, "_manifest.csv")
        if os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(header_line)

    if hdr_a:
        truncate_manifest(old_dir_a, hdr_a)
    if hdr_b:
        truncate_manifest(old_dir_b, hdr_b)

    return arr_new, new_tid


# ===== Example usage =====
if __name__ == "__main__":
    seq_name = "seq01"
    arr = np.load(f"tracklets_array/{seq_name}.npy")  # Load existing array
    tracklets_root = f"tracklets_vis/{seq_name}"      # Root folder for tracklets

    tid_a = 18
    tid_b = 20

    arr_new, new_tid = merge_tracklets(arr, tracklets_root, tid_a, tid_b)
    np.save(f"tracklets_array/{seq_name}.npy", arr_new)

    print("New array shape:", arr_new.shape)
    print("Merged into new tracklet ID:", new_tid)
