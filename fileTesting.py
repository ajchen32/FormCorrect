from flask import Flask, Response, request, jsonify, send_file
from flask_cors import CORS
import os
from videogen import createCorrectionVideo
import json
import traceback
import base64
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
        textCorrections, plot1, plot2, plot3 = createCorrectionVideo(file1_path, file2_path)
    except Exception as e:
        print(f"Error generating video: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Error generating video: {e}"}), 500
    with open(output_file_path, 'rb') as video_file:
        video_data = video_file.read()
    if(textCorrections == None):
        textCorrections = ["Failed to generate video"]
    # Send the generated video back to the caller
    video_base64 = base64.b64encode(video_data).decode('utf-8')
    plot1_base64 = base64.b64encode(plot1.read()).decode('utf-8')
    plot2_base64 = base64.b64encode(plot2.read()).decode('utf-8')
    plot3_base64 = base64.b64encode(plot3.read()).decode('utf-8')
    #print byte size of video_data for debugging
    print(f"Video data size: {len(video_data)} bytes")
    response = {
        "textCorrections": textCorrections,
        "videoBase64": video_base64,
        "videoMimeType": "video/mp4",
        "plots" : [
        {"imageBase64": plot1_base64, "mimeType": "image/png"},
        {"imageBase64": plot2_base64, "mimeType": "image/png"},
        {"imageBase64": plot3_base64, "mimeType": "image/png"}
    ]
    }
    return jsonify(response)


@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
