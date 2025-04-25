from flask import Flask, Response, request, jsonify, send_file
from flask_cors import CORS
import os
from videogen import createCorrectionVideo
import json
# Enable CORS for all routes
app = Flask(__name__)
CORS(app)

# Create an upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Route to handle file upload and video generation
@app.route('/upload', methods=['POST'])
def generate_video():
    # Check if exactly two files are provided
    if len(request.files) != 2:
        return jsonify({"error": "Exactly two files are required"}), 400

    # Get the files by their field names
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    # Validate the files
    if not file1 or file1.filename == '':
        return jsonify({"error": "First file is missing or has no filename"}), 400
    if not file2 or file2.filename == '':
        return jsonify({"error": "Second file is missing or has no filename"}), 400

    # Save the files to the server
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
    file1.save(file1_path)
    file2.save(file2_path)


    # Generate the video
    output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], file1.filename)
    textCorrections = ["Failed to generate video"]
    try:
        textCorrections = createCorrectionVideo(file1_path, file2_path)
    except Exception as e:
        return jsonify({"error": f"Error generating video: {e}"}), 500
    with open(output_file_path, 'rb') as video_file:
        video_data = video_file.read()
    textCorrections = ["Failed to generate video"]
    # Send the generated video back to the caller
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    response = Response(
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"textCorrections\"\r\n\r\n"
        f"{json.dumps(textCorrections)}\r\n"
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{file1.filename}\"\r\n"
        f"Content-Type: video/mp4\r\n\r\n"
        .encode('utf-8') + video_data + f"\r\n--{boundary}--\r\n".encode('utf-8'),
        content_type=f"multipart/form-data; boundary={boundary}"
    )
    return response


@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
