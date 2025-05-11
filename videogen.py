import cv2
import numpy as np
import os
from recursiveregressionmodel import actual_model
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
BG_COLOR = (192, 192, 192)
def createCorrectedSet(strings):
    #use body parts dict and look through the strings to find the body parts that are in the string
    #and return the set of body parts
    body_parts = {
        0: "nose",
        1: "left eye (inner)",
        2: "left eye",
        3: "left eye (outer)",
        4: "right eye (inner)",
        5: "right eye",
        6: "right eye (outer)",
        7: "left ear",
        8: "right ear",
        9: "mouth (left)",
        10: "mouth (right)",
        11: "left shoulder",
        12: "right shoulder",
        13: "left elbow",
        14: "right elbow",
        15: "left wrist",
        16: "right wrist",
        17: "left pinky",
        18: "right pinky",
        19: "left index",
        20: "right index",
        21: "left thumb",
        22: "right thumb",
        23: "left hip",
        24: "right hip",
        25: "left knee",
        26: "right knee",
        27: "left ankle",
        28: "right ankle",
        29: "left heel",
        30: "right heel",
        31: "left foot index",
        32: "right foot index",
    }
    correctedSet = set()
    for string in strings:
        for i in range(len(body_parts)):
            if body_parts[i] in string:
                correctedSet.add(i)
    return correctedSet
def createCorrectionVideo(file_path1, file_path2):



    cap = cv2.VideoCapture(file_path1)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame.")

    frame_height, frame_width, _ = frame.shape
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, os.path.basename(file_path1))

    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Alternative codec
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))
    strings = []
    frame_idx = 0
    strings, output, plot1, plot2, plot3 = actual_model(file_path2, file_path1)
    correctedSet = createCorrectedSet(strings)
    with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:

                    break    

#                     break

#                 if frame_idx >= len(x_cor) or frame_idx >= len(y_cor):
#                     print("Frame index exceeds coordinate array length. Stopping.")
#                     break

#                 x_cor_frame = np.array([x_cor[frame_idx]])
#                 y_cor_frame = np.array([y_cor[frame_idx]])
#                 x_goal_frame = np.array([x_goal[frame_idx]])
#                 y_goal_frame = np.array([y_goal[frame_idx]])

#                 try:
#                     output = actual_model_modified(world_coord_array1, world_coord_array2)
#                 except Exception as e:
#                     print(f"Error during regression: {e}")
#                 correctedSet = createCorrectedSet(output)
                # height, width, _ = frame.shape
                # for i in range(iterations):
                #     x_suggested = np.clip(int(x_cor_frame[0] * width), 0, width - 1)
                #     y_suggested = np.clip(int(y_cor_frame[0] * height), 0, height - 1)
                #     cv2.circle(frame, (x_suggested, y_suggested), 5, (0, 255, 0), -1)

                # x_original = np.clip(int(x_cor[frame_idx] * width), 0, width - 1)
                # y_original = np.clip(int(y_cor[frame_idx] * height), 0, height - 1)
                # cv2.circle(frame, (x_original, y_original), 5, (0, 0, 255), -1)
                

                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                annotated_image = frame.copy()
                # Draw segmentation on the image.
                # To improve segmentation around boundaries, consider applying a joint
                # bilateral filter to "results.segmentation_mask" with "image"
                
                
                condition = True #np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(frame.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
                annotated_image = np.where(condition, annotated_image, bg_image)
                filtered_connections = [
                    connection for connection in mp_pose.POSE_CONNECTIONS
                    if connection[0] in correctedSet or connection[1] in correctedSet
                ]
                #filter the landmarks to only show the ones in the corrected set
                filtered_landmarks = [
                    results.pose_landmarks.landmark[idx]
                    for idx in correctedSet if idx < len(results.pose_landmarks.landmark)
                ]
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                out.write(annotated_image)
                frame_idx += 1
        finally:
            cap.release()
            out.release()
            print(f"Video saved at {output_file_path}")
            return strings, plot1, plot2, plot3

# Example usage

# if __name__ == "__main__":
#     file_path1 = r"C:\Users\dhruv\OneDrive\Documents\GitHub\team-82-FormCorrect\model\WIN_20250404_16_16_29_Pro.mp4"
#     file_path2 = r"C:\Users\dhruv\OneDrive\Documents\GitHub\team-82-FormCorrect\model\WIN_20250404_16_16_40_Pro.mp4"
#     print(createCorrectionVideo(file_path1, file_path2))

# if(__name__ == "__main__"):
#     file_path1 = r"team-82-FormCorrect/uploads/SomeGuySquatting.mp4"
#     file_path2 = r"team-82-FormCorrect/uploads/BuffGuySquatting.mp4"
#     print(createCorrectionVideo(file_path1, file_path2))

if(__name__ == "__main__"):
    file_path1 = r"team-82-FormCorrect/uploads/GirlDancingGif.mp4"
    file_path2 = r"team-82-FormCorrect/uploads/GuyDancingGif.mp4"
    print(createCorrectionVideo(file_path1, file_path2))



