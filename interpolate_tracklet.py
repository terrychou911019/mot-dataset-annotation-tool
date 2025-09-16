import os
import cv2
import numpy as np

import os
import cv2
import numpy as np

def interpolate_tracklet(arr, img_dir, tracklets_root, tid, max_gap=10):
    """
    Interpolate missing bboxes for a tracklet within gaps <= max_gap.
    - arr: numpy array (num_tracklets+1, num_frames+1, 4), 1-based indexing
    - img_dir: folder containing per-frame images (000001.jpg, ...)
    - tracklets_root: root folder containing tracklet_xxxx directories
    - tid: tracklet id to interpolate
    - max_gap: maximum allowed gap size for interpolation (inclusive)

    Returns:
      arr_new: updated numpy array with interpolated bboxes filled in
    """
    arr_new = arr.copy()
    num_tracklets, num_frames, _ = arr_new.shape

    tracklet_dir = os.path.join(tracklets_root, f"tracklet_{tid:04d}")
    manifest_path = os.path.join(tracklet_dir, "_manifest.csv")

    # Load manifest if exists
    header = "frame,x,y,w,h,filename\n"
    lines = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines:
            header = lines[0]
            lines = lines[1:]
    manifest_entries = {int(l.split(",")[0]): l for l in lines}

    # Get all frames where tracklet has bbox
    frames = np.where(~np.isnan(arr_new[tid, :, 0]))[0]
    frames = frames[frames >= 1]  # skip index 0
    if len(frames) < 2:
        return arr_new  # nothing to interpolate

    for i in range(len(frames) - 1):
        t = frames[i]
        t_next = frames[i+1]
        gap = t_next - t
        
        if 1 < gap <= max_gap:
            bbox_start = arr_new[tid, t, :]
            bbox_end = arr_new[tid, t_next, :]
            print(f"Frame {t} to {t_next}, gap {gap}")

            for k in range(1, gap):
                f = t + k
                alpha = k / gap
                bbox_interp = (1 - alpha) * bbox_start + alpha * bbox_end
                arr_new[tid, f, :] = bbox_interp

                # Generate crop for interpolated bbox
                img_path = os.path.join(img_dir, f"{f:06d}.jpg")
                if not os.path.exists(img_path):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue

                H, W = img.shape[:2]
                x, y, w, h = bbox_interp
                x0 = int(np.floor(x))
                y0 = int(np.floor(y))
                x1 = int(np.ceil(x + w))
                y1 = int(np.ceil(y + h))
                x0 = max(0, min(x0, W - 1))
                y0 = max(0, min(y0, H - 1))
                x1 = max(0, min(x1, W))
                y1 = max(0, min(y1, H))

                if x1 <= x0 or y1 <= y0:
                    continue
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                out_name = f"{f:06d}.jpg"
                out_path = os.path.join(tracklet_dir, out_name)
                cv2.imwrite(out_path, crop)

                manifest_entries[f] = f"{f},{x0},{y0},{x1-x0},{y1-y0},{out_name}\n"

    # Rewrite manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(header)
        for frame_id in sorted(manifest_entries.keys()):
            f.write(manifest_entries[frame_id])

    return arr_new


if __name__ == "__main__":
    seq_name = "seq01"
    arr = np.load(f"tracklets_array/{seq_name}.npy")
    img_dir = f"dataset/{seq_name}/img1"
    tracklets_root = f"tracklets_vis/{seq_name}"

    tid = 25
    arr_new = interpolate_tracklet(arr, img_dir, tracklets_root, tid, max_gap=10)

    np.save(f"tracklets_array/{seq_name}_interpolate.npy", arr_new)