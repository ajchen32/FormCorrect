import cv2
import tensorflow as tf
import ffmpeg
import subprocess
import json
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#needed for the newer version of mediapipe

model_path = "model/pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

setup = PoseLandmarkerOptions(base_options = BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, num_poses=1, min_pose_detection_confidence = .5,
    min_pose_presence_confidence=.5, min_tracking_confidence=.5)
#necessary arguments that defines what aspect of the mode will be used

# print(cv2.__version__)
# print(mp.__version__)
# print(tf.__version__)
#test to check if the downloaded lib on the virtual env is working

#------was my debugging stuff 
# import os
# file_path = "C:/Users/soohw/Downloads/pose_test_vid.mp4.mp4"
# if os.path.exists(file_path):
#     print("File exists!")
# else:
#     print("File does not exist. Check the path.")
#------cause i didnt get the right path (dw about it :D)

# file_path = "C:/Users/soohw/Downloads/pose_test_vid.mp4"
# try:
#     probe = ffmpeg.probe(file_path)
#     video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
#     rotation = int(video_stream.get('tags', {}).get('rotate', 0))
# except Exception as e:
#     print(f"Error reading rotation metadata: {e}")
#     rotation = 0
#because opencv does not automatically support metadata that is given along with the video u take on your phone, opencv cannot 
#register the necessary rotations needed so im rotating it. (using ffmpeg to detect rotation metadata).

def get_metadata_rotation(file_path):
    try:
        # Use ffprobe to get rotation metadata
        command = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format_tags=rotation:stream_tags=rotate', '-of', 'json', file_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        metadata = json.loads(result.stdout)
        video_stream = next((stream for stream in metadata['streams'] if stream['codec_type'] == 'video'), None)
        rotation = int(video_stream.get('tags', {}).get('rotate', 0))
        return rotation
    except Exception as e:
        print(f"Error reading rotation metadata: {e}")
        return 0
    
def original_video_orientation(file_path):
        # Open the video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame.")

    # Get frame dimensions
    height, width, _ = frame.shape
    cap.release()

    # Determine orientation based on aspect ratio
    if height > width:
        return 0  
    else:
        return 90 
    #be careful the code does not support if it were to change the resolution mid video

def rotate(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame  # No rotation needed
    
def resize(frame, max_height, max_width):
    height, width = frame.shape[:2]
    ratio = width / height
    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(new_width / ratio)
        else:
            new_height = max_height
            new_width = int(new_height * ratio)

    frame = cv2.resize(frame, (new_width, new_height))
    return frame


def proccess_frame(file_path):
    rotation = get_metadata_rotation(file_path)

    print(rotation)
    # Step 2: If no metadata, use fallback logic
    if rotation == 0:
        rotation = original_video_orientation(file_path)
    print(rotation)
    # Step 3: Open the video and process frames
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    
    screen_width = 1280
    screen_height = 720
    world_coord_list = []
    with PoseLandmarker.create_from_options(setup) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate the frame
            frame = rotate(frame, rotation)
            frame = resize(frame, screen_height, screen_width)
            
            new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            #converts the opencv frame to mediapipe image object so we can apply necessary functions on it

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            #gets the timestamp for mediapipe

            pose_landmarker = landmarker.detect_for_video(new_image, timestamp_ms)
            #detects pose landmarks in this frame

            if pose_landmarker.pose_world_landmarks:
                #btw this library or this function can detect multiple poses
                #ie if there are multiple ppl in the frame
                #hence why we use the 0 index because we want the most prominent person's pose
                pose_world_landmarks = pose_landmarker.pose_world_landmarks[0]
                    #[
                    #        [ [x0, y0, z0], [x1, y1, z1], ..., [x32, y32, z32] ],  # Frame 0
                    #        [ [x0, y0, z0], [x1, y1, z1], ..., [x32, y32, z32] ],  # Frame 1
                    #        ...
                    #    ]
                world_coord_list.append([[landmark.x, landmark.y, landmark.z] for landmark in pose_world_landmarks])
                
                #for output printing purposes
                # frame_landmark = [[landmark.x, landmark.y, landmark.z] for landmark in pose_world_landmarks]
                # for landmark_idx, (x, y, z) in enumerate(frame_landmark):
                #     print(f"  Landmark {landmark_idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            
            world_coord_array = np.array(world_coord_list)

            # for frame_idx, frame_landmarks in enumerate(world_coord_array):
            #     print(f"Frame {frame_idx + 1}:")
            #     for landmark_idx, (x, y, z) in enumerate(frame_landmarks):
            #         print(f"  Landmark {landmark_idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            #         break
            #     print()  # Add a blank line between frames
            #dont run this T_T

            #if u do want to run any of these print statements u can press q to exit and it will still print a decent amount

            # Display the frame 
            cv2.imshow("Video", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
                break

    cap.release()
    cv2.destroyAllWindows()

proccess_frame("examples\WIN_20250326_15_39_18_Pro.mp4")




