import cv2
from PIL import Image
import os


def sample_frames(video_path, output_folder, frames_to_sample=4):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for i in range(frames_to_sample):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * (total_frames // frames_to_sample))
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            output_path = os.path.join(output_folder, f"{base_name}-{i + 1}.png")
            pil_image.save(output_path)

    video.release()


def main():
    input_folder = "/Users/dvd/Downloads/mime_test"
    output_folder = input_folder  # Save in the same folder

    for file in os.listdir(input_folder):
        if file.endswith(".mp4"):
            video_path = os.path.join(input_folder, file)
            sample_frames(video_path, output_folder)


if __name__ == "__main__":
    main()