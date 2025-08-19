from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image, ExifTags
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Add a context processor to inject url_prefix into all templates
@app.context_processor
def inject_url_prefix():
    return dict(url_prefix='/demo/ocr', api_prefix='/demo/ocr')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    api_files = []
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            api_files.append(('files', (filename, open(filepath, 'rb'), file.content_type)))

    try:
        # Send files to OCR API
        response = requests.post('http://localhost:8048/v1/ocr', files=api_files)

        # Clean up temporary files
        for _, (_, file_obj, _) in api_files:
            file_obj.close()
            os.remove(file_obj.name)

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'OCR service error'}), response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
