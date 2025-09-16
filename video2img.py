import argparse
from pathlib import Path
import cv2


def video2img(video_path: str, output_dir: str = "img1") -> int:
    """
    Convert a video into sequential images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the folder where frames will be saved.

    Returns:
        int: Number of frames (images) extracted.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        filename = f"{frame_idx:06d}.jpg"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), frame)
        frame_idx += 1

    cap.release()
    total_frames = frame_idx - 1
    print(f"Saved {total_frames} images to folder: {output_path}")
    return total_frames


def main():
    parser = argparse.ArgumentParser(description="Convert a video into images.")
    parser.add_argument("sequence", type=str, help="Sequence name (e.g., seq01).")
    parser.add_argument("--root", type=str, default="videos",
                        help="Folder that contains {sequence}.mp4 (default: videos)")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Folder to save {sequence}/img1 (default: dataset)")
    args = parser.parse_args()

    video_file = Path(args.root) / f"{args.sequence}.mp4"
    output_dir = Path(args.output_dir) / args.sequence / "img1"

    video2img(str(video_file), str(output_dir))


if __name__ == "__main__":
    main()
