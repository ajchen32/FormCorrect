import cv2
import tensorflow as tf
import ffmpeg
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import linalg as la

# from numpy import linalg

#r"C:\Users\dhruv\OneDrive\Documents\GitHub\team-82-FormCorrect\model\pose_landmarker_full.task"
model_path = r"team-82-FormCorrect\UTF-8pose_landmarker_heavy.task"


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

setup = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
# necessary arguments that defines what aspect of the mode will be used

# print(cv2.__version__)
# print(mp.__version__)
# print(tf.__version__)
# test to check if the downloaded lib on the virtual env is working

# ------was my debugging stuff
# import os
# file_path = "C:/Users/soohw/Downloads/pose_test_vid.mp4.mp4"
# if os.path.exists(file_path):
#     print("File exists!")
# else:
#     print("File does not exist. Check the path.")
# ------cause i didnt get the right path (dw about it :D)

# file_path = "C:/Users/soohw/Downloads/pose_test_vid.mp4"
# try:
#     probe = ffmpeg.probe(file_path)
#     video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
#     rotation = int(video_stream.get('tags', {}).get('rotate', 0))
# except Exception as e:
#     print(f"Error reading rotation metadata: {e}")
#     rotation = 0
# because opencv does not automatically support metadata that is given along with the video u take on your phone, opencv cannot
# register the necessary rotations needed so im rotating it. (using ffmpeg to detect rotation metadata).


def get_metadata_rotation(file_path):
    try:
        # Use ffprobe to get rotation metadata
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=rotation:stream_tags=rotate",
            "-of",
            "json",
            file_path,
        ]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        metadata = json.loads(result.stdout)
        video_stream = next(
            (
                stream
                for stream in metadata["streams"]
                if stream["codec_type"] == "video"
            ),
            None,
        )
        rotation = int(video_stream.get("tags", {}).get("rotate", 0))
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
    # be careful the code does not support if it were to change the resolution mid video


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
    new_width = 0
    new_height = 0
    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(new_width / ratio)
        else:
            new_height = max_height
            new_width = int(new_height * ratio)
    try:
        frame = cv2.resize(frame, (new_width, new_height))
        return frame
    except:
        return frame


def proccess_frame(file_path):
    rotation = get_metadata_rotation(file_path)
    graph = __name__ == "__main__"
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
    # world_coord_array = np.empty((0, 33, 3))
    world_coord_list = []
    edges_array = []
    with PoseLandmarker.create_from_options(setup) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate the frame
            #frame = rotate(frame, rotation)
            frame = resize(frame, screen_height, screen_width)

            new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # converts the opencv frame to mediapipe image object so we can apply necessary functions on it

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            # gets the timestamp for mediapipe

            pose_landmarker = landmarker.detect_for_video(new_image, timestamp_ms)
            #detects pose landmarks in this frame
            if graph:
                ax.cla()
            if pose_landmarker.pose_world_landmarks:
                # btw this library or this function can detect multiple poses
                # ie if there are multiple ppl in the frame
                # hence why we use the 0 index because we want the most prominent person's pose
                pose_world_landmarks = pose_landmarker.pose_world_landmarks[0]
                # [
                #        [ [x0, y0, z0], [x1, y1, z1], ..., [x32, y32, z32] ],  # Frame 0
                #        [ [x0, y0, z0], [x1, y1, z1], ..., [x32, y32, z32] ],  # Frame 1
                #        ...
                #    ]
                # frame_landmark = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_world_landmarks])
                # world_coord_array = np.vstack([world_coord_array, frame_landmark[np.newaxis, :, :]])
                # I changed the code using np.vstack to now create a 3d array that stores number of frames x (xyz) x 33 points

                # latest version
                frame_landmark = np.array(
                    [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in pose_world_landmarks
                    ]
                )
                world_coord_list.append(frame_landmark)
                # latest version

                # for output printing purposes
                # frame_landmark = [[landmark.x, landmark.y, landmark.z] for landmark in pose_world_landmarks]
                # for landmark_idx, (x, y, z) in enumerate(frame_landmark):
                #     print(f"  Landmark {landmark_idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
                if graph:
                    ax.scatter(frame_landmark[:, 0], frame_landmark[:, 1], frame_landmark[:, 2], c="c", marker="o")
                frame_edges = []
                for connection in mp.solutions.pose.POSE_CONNECTIONS:
                    idx1, idx2 = connection
                    x_vals = [frame_landmark[idx1, 0], frame_landmark[idx2, 0]]
                    y_vals = [frame_landmark[idx1, 1], frame_landmark[idx2, 1]]
                    z_vals = [frame_landmark[idx1, 2], frame_landmark[idx2, 2]]
                    if graph:
                        ax.plot(x_vals, y_vals, z_vals, "b", linewidth = 2)
                    point1 = frame_landmark[idx1]
                    point2 = frame_landmark[idx2]
                    frame_edges.append([point1, point2])
                edges_array.append(np.array(frame_edges))
            if graph:
                plt.draw()
                plt.pause(0.01)


            # for frame_idx, frame_landmarks in enumerate(world_coord_array):
            #     print(f"Frame {frame_idx + 1}:")
            #     for landmark_idx, (x, y, z) in enumerate(frame_landmarks):
            #         print(f"  Landmark {landmark_idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            #         break
            #     print()  # Add a blank line between frames
            # dont run this T_T

            # if u do want to run any of these print statements u can press q to exit and it will still print a decent amount

            # Display the frame 
            if graph:
                cv2.imshow("Video", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
                    break
    world_coord_array = np.array(world_coord_list)
    edges_array = np.array(edges_array)
    print("Edge array shape:", edges_array.shape)
    # np.save("pose_edges.npy", edges_array)
    cap.release()
    cv2.destroyAllWindows()
    return world_coord_array, edges_array

def process_frame_without_video_output(file_path): # just to leave original unmodified, processes the frames without the video output to increase speed
    rotation = get_metadata_rotation(file_path)
    if rotation == 0:
        rotation = original_video_orientation(file_path)
    # Step 3: Open the video and process frames
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    screen_width = 1280
    screen_height = 720
    # world_coord_array = np.empty((0, 33, 3))
    world_coord_list = []
    edges_array = []
    with PoseLandmarker.create_from_options(setup) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate the frame
            #frame = rotate(frame, rotation)
            frame = resize(frame, screen_height, screen_width)

            new_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # converts the opencv frame to mediapipe image object so we can apply necessary functions on it

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            # gets the timestamp for mediapipe

            pose_landmarker = landmarker.detect_for_video(new_image, timestamp_ms)
            # detects pose landmarks in this frame
            ax.cla()
            if pose_landmarker.pose_world_landmarks:
                # btw this library or this function can detect multiple poses
                # ie if there are multiple ppl in the frame
                # hence why we use the 0 index because we want the most prominent person's pose
                pose_world_landmarks = pose_landmarker.pose_world_landmarks[0]
                # [
                #        [ [x0, y0, z0], [x1, y1, z1], ..., [x32, y32, z32] ],  # Frame 0
                #        [ [x0, y0, z0], [x1, y1, z1], ..., [x32, y32, z32] ],  # Frame 1
                #        ...
                #    ]
                # frame_landmark = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_world_landmarks])
                # world_coord_array = np.vstack([world_coord_array, frame_landmark[np.newaxis, :, :]])
                # I changed the code using np.vstack to now create a 3d array that stores number of frames x (xyz) x 33 points

                # latest version
                frame_landmark = np.array(
                    [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in pose_world_landmarks
                    ]
                )
                world_coord_list.append(frame_landmark)
                # latest version

                # for output printing purposes
                # frame_landmark = [[landmark.x, landmark.y, landmark.z] for landmark in pose_world_landmarks]
                # for landmark_idx, (x, y, z) in enumerate(frame_landmark):
                #     print(f"  Landmark {landmark_idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

                ax.scatter(
                    frame_landmark[:, 0],
                    frame_landmark[:, 1],
                    frame_landmark[:, 2],
                    c="c",
                    marker="o",
                )
                frame_edges = []
                for connection in mp.solutions.pose.POSE_CONNECTIONS:
                    idx1, idx2 = connection
                    x_vals = [frame_landmark[idx1, 0], frame_landmark[idx2, 0]]
                    y_vals = [frame_landmark[idx1, 1], frame_landmark[idx2, 1]]
                    z_vals = [frame_landmark[idx1, 2], frame_landmark[idx2, 2]]
                    ax.plot(x_vals, y_vals, z_vals, "b", linewidth=2)
                    point1 = frame_landmark[idx1]
                    point2 = frame_landmark[idx2]
                    frame_edges.append([point1, point2])
                edges_array.append(np.array(frame_edges))
            
    world_coord_array = np.array(world_coord_list)
    edges_array = np.array(edges_array)
    print("Edge array shape:", edges_array.shape)
    # np.save("pose_edges.npy", edges_array)
    cap.release()
    cv2.destroyAllWindows()
    return world_coord_array, edges_array

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")
ax.set_title("BlazePose 3D World Landmarks (Tasks API)")
if(__name__ == "__main__"):
    world_array, edge_array = proccess_frame("c:/Users/soohw/Downloads/pose_test_2.mp4")




def covaraince_matrix(video_file):
    world_array, edge_array = proccess_frame(video_file)

# world array is (num_Frames, 33, 3)
# I need it to be by number of (keypoints, data) ie flatten the keypoints and insert tuples?
# nah tuples is better for visualization and actual working with data but covaraince is best done accross each axis

    world_array_x = world_array[:, :, 0]
    world_array_y = world_array[:, :, 1]
    world_array_z = world_array[:, :, 2]
    # (num_frames, 33)

    num_of_frames = world_array_x.shape[0]
    num_of_points = world_array_x.shape[1]

    print(num_of_frames)
    print(num_of_points)

    average_keypoints = np.mean(world_array, axis=0)


    world_array_x_centered = world_array_x - average_keypoints[:, 0]
    world_array_y_centered = world_array_y - average_keypoints[:, 1]
    world_array_z_centered = world_array_z - average_keypoints[:, 2]
    print(np.mean(world_array_x_centered, axis=0))

    cov_matrix_x = np.cov(world_array_x_centered, rowvar=False)
    cov_matrix_y = np.cov(world_array_y_centered, rowvar=False)
    cov_matrix_z = np.cov(world_array_z_centered, rowvar=False)
    return cov_matrix_x, cov_matrix_y, cov_matrix_z




def main():
    #choose one
    video_file = "team-82-FormCorrect\SomeGuySquatting.mp4"
    #video_file = "team-82-FormCorrect\GuyMovingArms.mp4"

    #world_coord_array, edges_array = proccess_frame(video_file)
    world_coord_array, edges_array = process_frame_without_video_output(video_file)

    print(world_coord_array)
    print(edges_array)

    # you have to wait a bit for it to finish

if __name__ == "__main__":
    main()
