import cv2
import torch


def sample_frames(video_path, num_frames=4, start=0, end=None) -> torch.Tensor:
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Convert start and end from seconds to frame numbers
    start_frame = int(start * fps)
    if end is None:
        end_frame = frame_count
    else:
        end_frame = int(end * fps)

    # Limit the end frame to the total number of frames in the video
    end_frame = min(end_frame, frame_count)

    frames = []
    # Calculate the number of frames to skip to evenly sample num_frames between start_frame and end_frame
    step = (end_frame - start_frame) // num_frames
    for i in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert color from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if len(frames) == num_frames:
            break

    # Release the video capture object
    cap.release()

    # Convert the list of frames to a tensor
    frames = torch.tensor(frames)
    return frames
