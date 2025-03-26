import cv2

def draw_mismatched_boxes(image1_path, image2_path, joints_dict1, joints_dict2, output_path):
    # Load the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        raise ValueError("One or both images could not be loaded. Check the file paths.")

    # Iterate through the joints and compare bounding boxes
    for joint, box1 in joints_dict1.items():
        if joint in joints_dict2:
            box2 = joints_dict2[joint]
            # Draw the supposed bounding box from the first image (green box)
            x1, y1, w1, h1 = box1
            cv2.rectangle(image2, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)  # Green box for supposed position
            cv2.putText(image2, f"{joint} (expected)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Check if the bounding boxes do not match
            if box1 != box2:
                # Draw the mismatched box on the second image (red box)
                x2, y2, w2, h2 = box2
                cv2.rectangle(image2, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)  # Red box for mismatch
                cv2.putText(image2, f"{joint} (actual)", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            print(f"Joint '{joint}' not found in the second dictionary.")

    # Save the output image
    cv2.imwrite(output_path, image2)
    print(f"Output image with mismatched boxes saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Example dictionaries of joints and bounding boxes
    joints_dict1 = {
        #"joint1": (50, 50, 100, 100),
        "joint2": (700, 200, 200, 200),
    }
    joints_dict2 = {
        #"joint1": (50, 50, 100, 100),
        "joint2": (1300, 200, 200, 200),  # Mismatch
    }

    # Paths to the input images and output image
    image1_path = "examples\handIn.jpg"
    image2_path = "examples\handOut.jpg"
    output_path = "examples\output.jpg"

    draw_mismatched_boxes(image1_path, image2_path, joints_dict1, joints_dict2, output_path)