import os
import requests

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

    # Step 2: Send MP4 files to the upload endpoint
    upload_url = "http://localhost:5000/upload"
    model_dir = "./model"
    files = [f for f in os.listdir(model_dir) if f.endswith(".mp4")]

    if not files:
        print("No MP4 files found in the model directory.")
        return

    for file_name in files:
        file_path = os.path.join(model_dir, file_name)
        try:
            with open(file_path, "rb") as f:
                files_payload = {"file": (file_name, f, "video/mp4")}
                upload_response = requests.post(upload_url, files=files_payload)
                if upload_response.status_code == 200:
                    print(f"Successfully uploaded {file_name}: {upload_response.json()}")
                else:
                    print(f"Failed to upload {file_name}: {upload_response.status_code}, {upload_response.text}")
        except Exception as e:
            print(f"Error uploading {file_name}: {e}")

if __name__ == "__main__":
    main()