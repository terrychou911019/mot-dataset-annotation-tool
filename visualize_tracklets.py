from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2


def visualize_tracklets(arr: np.ndarray, resize_w: int, resize_h: int, img_dir: str, tracklets_root: str) -> None:
    """
    Create a folder per tracklet and save cropped bbox images (resized to resize_w*resize_h).

    Args:
        arr (np.ndarray): Tracklet array of shape (num_tracks+1, num_frames+1, 4).
        resize_w (int): Width to resize cropped bbox images.
        resize_h (int): Height to resize cropped bbox images.
        img_dir (str): Directory containing original frame images named as 000001.jpg, 000002.jpg, ...
        tracklets_root (str): Root directory to save per-tracklet folders.
    """
    tracklets_root_path = Path(tracklets_root)
    tracklets_root_path.mkdir(parents=True, exist_ok=True)

    JPEG_QUALITY = 95
    num_tracklets, num_frames, _ = arr.shape
    img_dir_path = Path(img_dir)

    for fid in range(1, num_frames):
        img_path = img_dir_path / f"{fid:06d}.jpg"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        H, W = img.shape[:2]

        # Iterate over all tracklets for this frame
        for tid in range(1, num_tracklets):
            bbox = arr[tid, fid]
            if np.any(np.isnan(bbox)):
                continue  # no bbox for this (tid, fid)

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

            # Resize to fixed size
            resized = cv2.resize(crop, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

            # Ensure tracklet folder exists
            tracklet_dir = tracklets_root_path / f"tracklet_{tid:04d}"
            tracklet_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{fid:06d}.jpg"
            out_path = tracklet_dir / out_name
            cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    print(f"All crops ({resize_w}*{resize_h}) saved under: {tracklets_root_path}")


def main():
    parser = argparse.ArgumentParser(description="Crop and save per-tracklet bbox images (resize_w*resize_h).")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("--arr_dir", type=str, default="tracklets_array",
                        help="Folder containing {sequence}.npy (default: tracklets_array).")
    parser.add_argument("--output_dir", type=str, default="tracklets_vis",
                        help="Folder to save crops under {sequence}/ (default: tracklets_vis).")
    args = parser.parse_args()

    arr_path = Path(args.arr_dir) / f"{args.sequence}.npy"
    arr = np.load(arr_path)
    resize_w, resize_h = 240, 480
    tracklets_root = Path(args.output_dir) / args.sequence
    img_dir = Path("dataset") / args.sequence / "img1"

    visualize_tracklets(arr, resize_w, resize_h, str(img_dir), str(tracklets_root))


if __name__ == "__main__":
    main()
