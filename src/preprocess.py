import os
import cv2
from config import DATA_DIR

def extract_frames_from_all_videos(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    videos = [file for file in os.listdir(source_folder) if file.endswith(".mp4")]
    for video_file in videos:
        video_path = os.path.join(source_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        success, image = cap.read()
        count = 0
        while success:
            frame_path = os.path.join(target_folder, f"{video_file[:-4]}_frame{count}.jpg")
            cv2.imwrite(frame_path, image)
            success, image = cap.read()
            count += 1
        cap.release()
