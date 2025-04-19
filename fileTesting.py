from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from pose import proccess_frame
#getting the algorthim to extract points and edges; #returns 3d numpy array of points and then 4d for edges

# Enable CORS for all routes
app = Flask(__name__)
CORS(app)

# Create an upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
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

    #Commented these two things out since it wasn't working. File uploads correctly but something broke here. - CJ
    # world_coord_array, edges_array = proccess_frame(file1_path)
    # world_coord_array_1, edges_array_1 = proccess_frame(file2_path)

    return jsonify({
        "message": "Files uploaded successfully",
        "files": [file1.filename, file2.filename]
    }), 200

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
