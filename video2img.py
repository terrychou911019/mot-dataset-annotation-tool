import cv2
import os

def video2img(video_path, output_folder="img1"):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open video:", video_path)
        return

    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        # Format filename: 000001.jpg, 000002.jpg ...
        filename = f"{frame_idx:06d}.jpg"
        filepath = os.path.join(output_folder, filename)

        # Save the image
        cv2.imwrite(filepath, frame)
        frame_idx += 1

    cap.release()
    print(f"Done! Saved {frame_idx - 1} images to folder: {output_folder}")

if __name__ == "__main__":
    seq_name = 'seq01'
    video_file = rf'videos/{seq_name}.mp4' 
    output_folder = rf'dataset/{seq_name}/img1'
    
    video2img(video_file, output_folder)
