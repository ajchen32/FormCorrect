import os
import requests
import base64
import json

def main():
    # Step 1: Test if the server is running
    test_url = "http://localhost:5000/test"
    try:
        response = requests.get(test_url)
        if response.status_code == 200 and response.json() == {"message": "Test endpoint"}:
            print("Test endpoint is working.")
        else:
            print("Unexpected response from test endpoint:", response.json())
            return
    except requests.exceptions.RequestException as e:
        print("Error connecting to the test endpoint:", e)
        return

    # Step 2: Send two MP4 files to the upload endpoint
    upload_url = "http://localhost:5000/upload"
    model_dir = "./model"
    files = [f for f in os.listdir(model_dir) if f.endswith(".mp4")]

    if len(files) < 2:
        print("Not enough MP4 files found in the model directory. At least two are required.")
        return

    # Select the first two files
    file1_path = os.path.join(model_dir, files[0])
    file2_path = os.path.join(model_dir, files[1])

    try:
        with open(file1_path, "rb") as f1, open(file2_path, "rb") as f2:
            files_payload = {
                "file1": (files[0], f1, "video/mp4"),
                "file2": (files[1], f2, "video/mp4"),
            }
            upload_response = requests.post(upload_url, files=files_payload)

            if upload_response.status_code == 200:
                # Parse the JSON response
                response_data = upload_response.json()
                text_corrections = response_data.get("textCorrections", [])
                video_base64 = response_data.get("videoBase64", "")

                # Print the text corrections
                print("Text Corrections:")
                for correction in text_corrections:
                    print(f"- {correction}")

                # Decode and save the video
                if video_base64:
                    video_data = base64.b64decode(video_base64)
                    output_file_path = "output_file.mp4"
                    with open(output_file_path, "wb") as output_file:
                        output_file.write(video_data)
                    print(f"Video saved as {output_file_path}")
                else:
                    print("No video data found in the response.")
            else:
                print(f"Failed to upload files: {upload_response.status_code}, {upload_response.text}")
    except Exception as e:
        print(f"Error uploading files: {e}")

if __name__ == "__main__":
    main()