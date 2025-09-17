from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2


def interpolate_tracklet(arr: np.ndarray, tid: int, max_gap: int, img_dir: str, tracklets_root: str) -> np.ndarray:
    """
    Linearly interpolate missing bboxes for a single tracklet within gaps <= max_gap.

    Args:
        arr (np.ndarray): Tracklet array of shape (num_tracks+1, num_frames+1, 4).
        tid (int): Tracklet ID to interpolate (1-based).
        max_gap (int): Maximum allowed gap length (inclusive) for interpolation.
        img_dir (str): Directory containing per-frame images named as 000001.jpg, 000002.jpg, ...
        tracklets_root (str): Root folder containing per-tracklet folders named as tracklet_XXXX.

    Returns:
        arr_new (np.ndarray): Updated array with interpolated bboxes filled in for gaps <= max_gap.
    """
    num_tracks, num_frames, _ = arr.shape
    if not (1 <= tid < num_tracks):
        raise ValueError(f"tid out of range: {tid} (valid: 1..{num_tracks-1})")

    arr_new = arr.copy()
    tracklet_dir = Path(tracklets_root) / f"tracklet_{tid:04d}"
    tracklet_dir.mkdir(parents=True, exist_ok=True)

    img_dir_path = Path(img_dir)

    # Frames where this tracklet has a bbox (ignore padding index 0)
    frames = np.where(~np.isnan(arr_new[tid, :, 0]))[0]
    frames = frames[frames >= 1]
    if frames.size < 2:
        return arr_new  # nothing to interpolate

    for i in range(frames.size - 1):
        t = int(frames[i])
        t_next = int(frames[i + 1])
        gap = t_next - t

        # Only interpolate for gaps 2..max_gap (strictly missing in-between)
        if 1 < gap <= max_gap:
            bbox_start = arr_new[tid, t, :].astype(float)
            bbox_end = arr_new[tid, t_next, :].astype(float)

            print(f"Interpolating tracklet {tid} from frame {t} to {t_next} (gap={gap})")

            for k in range(1, gap):
                f = t + k
                alpha = k / gap
                bbox_interp = (1.0 - alpha) * bbox_start + alpha * bbox_end
                arr_new[tid, f, :] = bbox_interp

                # Save cropped image for the interpolated bbox if the frame exists on disk
                img_path = img_dir_path / f"{f:06d}.jpg"
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                H, W = img.shape[:2]
                x, y, w, h = bbox_interp
                x0 = int(np.floor(x))
                y0 = int(np.floor(y))
                x1 = int(np.ceil(x + w))
                y1 = int(np.ceil(y + h))

                # Clamp to image bounds
                x0 = max(0, min(x0, W - 1))
                y0 = max(0, min(y0, H - 1))
                x1 = max(0, min(x1, W))
                y1 = max(0, min(y1, H))

                # Skip invalid or empty crops after clamping
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                out_path = tracklet_dir / f"{f:06d}.jpg"
                cv2.imwrite(str(out_path), crop)

    return arr_new


def main():
    parser = argparse.ArgumentParser(description="Interpolate missing bboxes for a tracklet (gap <= max_gap).")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("tracklet_id", type=int, help="Tracklet ID to interpolate (1-based).")
    parser.add_argument("--max_gap", type=int, default=10, help="Maximum gap length to interpolate (inclusive).")
    parser.add_argument("--arr_dir", type=str, default="tracklets_array",
                        help="Folder containing {sequence}.npy (default: tracklets_array).")
    parser.add_argument("--output_dir", type=str, default="tracklets_vis",
                        help="Root folder containing per-sequence tracklets (default: tracklets_vis).")
    args = parser.parse_args()

    arr_path = Path(args.arr_dir) / f"{args.sequence}.npy"
    arr = np.load(arr_path)
    img_dir = Path("dataset") / args.sequence / "img1"
    tracklets_root = Path(args.output_dir) / args.sequence

    arr_new = interpolate_tracklet(
        arr=arr,
        tid=args.tracklet_id,
        max_gap=args.max_gap,
        img_dir=str(img_dir),
        tracklets_root=str(tracklets_root),
    )

    # Overwrite the same npy 
    np.save(arr_path, arr_new)

    print(f"Interpolated tracklet {args.tracklet_id:04d} (max_gap={args.max_gap}).")
    print(f"New array shape: {arr_new.shape}")


if __name__ == "__main__":
    # Example: python interpolate_tracklet.py seq01 25
    main()
