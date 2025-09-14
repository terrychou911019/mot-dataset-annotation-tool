import os
import cv2
import numpy as np
import csv

def visualize_tracklets(img_dir, arr, out_root):
    """
    Create a folder per tracklet and save cropped bbox images (resized to 240x480).
    Assumes:
      - arr shape: (num_tracklets+1, num_frames+1, 4), 1-based indexing.
      - arr[tid, fid] = (x, y, w, h) in pixel coordinates (may be float) or NaNs if missing.
      - Images in img_dir are named as 000001.jpg, 000002.jpg, ... (1-based frame IDs).
    Boundary checks:
      - Clamp bbox to image bounds.
      - Skip degenerate boxes after clamping.
    Output:
      - out_root/tracklet_0001/000001.jpg (resized to 240x480), ...
      - out_root/tracklet_0001/_manifest.csv (frame,x,y,w,h,filename,orig_w,orig_h)
    """
    os.makedirs(out_root, exist_ok=True)

    JPEG_QUALITY = 80

    num_tracklets, num_frames, _ = arr.shape

    # Lazy-open per-tracklet manifest files
    manifest_files = {}
    manifest_writers = {}

    def ensure_tracklet_dir_and_manifest(tid):
        """Ensure tracklet folder and manifest CSV are created and opened."""
        if tid not in manifest_writers:
            tracklet_dir = os.path.join(out_root, f"tracklet_{tid:04d}")
            os.makedirs(tracklet_dir, exist_ok=True)
            manifest_path = os.path.join(tracklet_dir, "_manifest.csv")
            f = open(manifest_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(["frame", "x", "y", "w", "h", "filename"])
            manifest_files[tid] = f
            manifest_writers[tid] = writer
        return os.path.join(out_root, f"tracklet_{tid:04d}"), manifest_writers[tid]

    target_w, target_h = 240, 480  # fixed output size

    for fid in range(1, num_frames):  # 1-based frames (skip index 0)
        img_path = os.path.join(img_dir, f"{fid:06d}.jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        # Iterate all tracklets for this frame
        for tid in range(1, num_tracklets):  # 1-based tracklet ids (skip index 0)
            bbox = arr[tid, fid]
            if np.any(np.isnan(bbox)):
                continue  # no bbox for (tid, fid)

            # Convert bbox to float then to int bounds (floor/ceil)
            x, y, w, h = bbox.astype(float)
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = int(np.ceil(x + w))
            y1 = int(np.ceil(y + h))

            # Clamp to image bounds
            x0 = max(0, min(x0, W - 1))
            y0 = max(0, min(y0, H - 1))
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))

            # Validate after clamping
            if x1 <= x0 or y1 <= y0:
                continue

            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            # Choose interpolation: INTER_AREA is generally safer for downscaling;
            # it is also acceptable for upscaling if you want a single policy.
            resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Ensure tracklet directory & manifest
            tracklet_dir, writer = ensure_tracklet_dir_and_manifest(tid)
            out_name = f"{fid:06d}.jpg"
            out_path = os.path.join(tracklet_dir, out_name)

            cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            # cv2.imwrite(out_path, resized)

            # Write manifest: store original bbox (int-rounded) + original crop size
            writer.writerow([
                fid,
                int(x0), int(y0), int(x1 - x0), int(y1 - y0),
                out_name,
            ])

    # Close all manifests
    for f in manifest_files.values():
        f.close()

    print(f"All crops (240x480) saved under: {out_root}")


# ===== Example usage =====
if __name__ == "__main__":
    # Example paths
    seq_name = "seq01"
    img_dir = f"dataset/{seq_name}/img1"           # per-frame images (000001.jpg ...)
    out_root = f"tracklets_vis/{seq_name}"
    arr = np.load(f"tracklets_array/{seq_name}.npy")  # (num_tracks+1, num_frames+1, 4), 1-based

    visualize_tracklets(img_dir, arr, out_root)
