from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
# from videogen import createCorrectionVideo

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

    # # Generate the video (Commented out for now to test, also commented out import cus it broke server - CJ)
    # output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_video.mp4')
    # try:
    #     createCorrectionVideo(file1_path, file2_path)
    # except Exception as e:
    #     return jsonify({"error": f"Error generating video: {e}"}), 500

    # # Send the generated video back to the caller
    # return send_file(output_file_path, as_attachment=True)

    return jsonify({"message": "Test endpoint"}), 200

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
