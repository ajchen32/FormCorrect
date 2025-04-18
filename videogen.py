import cv2
import numpy as np
import os
from pose import proccess_frame
from RegressionModel.regressionmodel import run_regression, calc_dist

def createCorrectionVideo(file_path1, file_path2):
    world_coord_array1, _ = proccess_frame(file_path1)
    world_coord_array2, _ = proccess_frame(file_path2)

    joint_idx = 15
    x_goal = world_coord_array2[:, joint_idx, 0]
    y_goal = world_coord_array2[:, joint_idx, 1]
    x_cor = world_coord_array1[:, joint_idx, 0]
    y_cor = world_coord_array1[:, joint_idx, 1]

    iterations = 10
    change_array = np.array([10, 10], dtype=float)

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

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Alternative codec
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= len(x_cor) or frame_idx >= len(y_cor):
                print("Frame index exceeds coordinate array length. Stopping.")
                break

            x_cor_frame = np.array([x_cor[frame_idx]])
            y_cor_frame = np.array([y_cor[frame_idx]])
            x_goal_frame = np.array([x_goal[frame_idx]])
            y_goal_frame = np.array([y_goal[frame_idx]])

            try:
                run_regression(x_goal_frame, y_goal_frame, x_cor_frame, y_cor_frame, iterations, change_array)
            except Exception as e:
                print(f"Error during regression: {e}")

            height, width, _ = frame.shape
            for i in range(iterations):
                x_suggested = np.clip(int(x_cor_frame[0] * width), 0, width - 1)
                y_suggested = np.clip(int(y_cor_frame[0] * height), 0, height - 1)
                cv2.circle(frame, (x_suggested, y_suggested), 5, (0, 255, 0), -1)

            x_original = np.clip(int(x_cor[frame_idx] * width), 0, width - 1)
            y_original = np.clip(int(y_cor[frame_idx] * height), 0, height - 1)
            cv2.circle(frame, (x_original, y_original), 5, (0, 0, 255), -1)

            out.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        out.release()
        print(f"Video saved at {output_file_path}")

# Example usage
file_path1 = r"C:\Users\dhruv\OneDrive\Documents\GitHub\team-82-FormCorrect\model\WIN_20250404_16_16_29_Pro.mp4"
file_path2 = r"C:\Users\dhruv\OneDrive\Documents\GitHub\team-82-FormCorrect\model\WIN_20250404_16_16_40_Pro.mp4"
createCorrectionVideo(file_path1, file_path2)



