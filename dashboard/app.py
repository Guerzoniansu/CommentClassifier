import os
import io
from flask import Flask, render_template, request, jsonify, send_file, session
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from extern import helper


app = Flask(__name__)
app.secret_key = '1234'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

classified_data_cache = {}


# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def load_models():
    return render_template('models.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_file_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0 Bytes"
    
    size_names = ["Bytes", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

@app.route('/classifier')
def load_classifier():
    return render_template('classifier.html')

@app.route('/preview_dataset', methods=['POST'])
def preview_dataset():
    """Handle dataset preview requests"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            })
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            })
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not supported. Please upload CSV, XLS, or XLSX files'
            })
        
        # Get file size for info
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        # Read file content
        file_content = file.read()
        file_stream = io.BytesIO(file_content)
        
        # Determine file type and read accordingly
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        try:
            if file_extension == 'csv':
                # Try different encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    try:
                        file_stream.seek(0)
                        df = pd.read_csv(file_stream, encoding=encoding, nrows=10)
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                
                if df is None:
                    return jsonify({
                        'success': False,
                        'error': 'Unable to read CSV file. Please check file encoding or format'
                    })
                    
            elif file_extension in ['xlsx', 'xls']:
                file_stream.seek(0)
                df = pd.read_excel(file_stream, nrows=10)
            else:
                return jsonify({
                    'success': False,
                    'error': 'Unsupported file format'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error reading file: {str(e)}'
            })
        
        # Get total number of rows (read full file for count)
        try:
            file_stream.seek(0)
            if file_extension == 'csv':
                # Use same encoding that worked for preview
                for encoding in encodings:
                    try:
                        file_stream.seek(0)
                        full_df = pd.read_csv(file_stream, encoding=encoding)
                        total_rows = len(full_df)
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
            else:
                file_stream.seek(0)
                full_df = pd.read_excel(file_stream)
                total_rows = len(full_df)
        except:
            total_rows = len(df)  # Fallback to preview rows if full read fails
        df = df.fillna('')
        columns = df.columns.tolist()
        rows = df.to_dict('records')
        
        # Prepare response data
        preview_data = {
            'columns': columns,
            'rows': rows
        }
        
        info_data = {
            'total_rows': total_rows,
            'file_size': format_file_size(file_size),
            'filename': filename
        }
        
        return jsonify({
            'success': True,
            'preview': preview_data,
            'info': info_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        })

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Handle full dataset upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            })
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            })
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not supported'
            })
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the uploaded file
        try:
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            
            # Here you can add your dataset processing logic
            # For example: data cleaning, feature extraction, etc.
            
            return jsonify({
                'success': True,
                'message': f'Dataset uploaded successfully. Shape: {df.shape}',
                'filename': filename,
                'rows': df.shape[0],
                'columns': df.shape[1]
            })
            
        except Exception as e:
            # Clean up file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                'success': False,
                'error': f'Error processing file: {str(e)}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        })


@app.route('/statistics', methods=['POST'])
def generate_statistics():
    """Process uploaded files and generate statistics"""
    try:
        file_count = int(request.form.get('file_count', 0))
        results = []
        
        for i in range(1, file_count + 1):
            file_key = f'file_{i}'
            sheet_key = f'sheet_{i}'
            column_key = f'column_{i}'
            model_key = f'model_{i}'
            
            if file_key not in request.files:
                continue
                
            file = request.files[file_key]
            sheet_name = request.form.get(sheet_key, '')
            column_name = request.form.get(column_key, '')
            model_id = request.form.get(model_key, '1')
            
            try:
                # Process single file
                result = process_single_file(file, sheet_name, column_name, model_id)
                results.append(result)
                
            except Exception as e:
                results.append({
                    'success': False,
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Processing failed: {str(e)}'
        })
    
def convert_numpy(obj):
    """Recursively convert NumPy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

def process_single_file(file, sheet_name, column_name, model_id):
    """Process a single file and return statistics"""
    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()

    cache_key = f"{filename}_{model_id}_{hash(column_name)}"
    
    # Read file content
    file_content = file.read()
    file_stream = io.BytesIO(file_content)
    
    # Load dataframe based on file type
    try:
        if file_extension == 'csv':
            # For CSV, sheet_name is ignored
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    file_stream.seek(0)
                    df = pd.read_csv(file_stream, encoding=encoding)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
                    
            if df is None:
                raise Exception("Could not read CSV file with any encoding")
                
        elif file_extension in ['xlsx', 'xls']:
            file_stream.seek(0)
            
            # Check if sheet exists
            excel_file = pd.ExcelFile(file_stream)
            available_sheets = excel_file.sheet_names
            
            if sheet_name not in available_sheets:
                raise Exception(f"Sheet '{sheet_name}' not found. Available sheets: {', '.join(available_sheets)}")
            
            df = pd.read_excel(file_stream, sheet_name=sheet_name)
            
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")
    
    # Validate column exists
    if column_name not in df.columns:
        available_columns = ', '.join(df.columns.tolist())
        raise Exception(f"Column '{column_name}' not found. Available columns: {available_columns}")
    
    # Generate statistics
    df = helper.clean_dataframe(df, column_name)
    stats = helper.generate_text_statistics(df, column_name)
    top_words = helper.get_word_frequency_data(df, column_name)
    
    # Generate charts (you can implement this based on your needs)
    charts, rec = helper.generate_charts_html_v2(df, column_name, model_id, cache_key)
    classified_data_cache[cache_key] = rec
    
    return convert_numpy({
        'success': True,
        'filename': filename,
        'stats': stats,
        'top_words': top_words,
        'charts': charts,
        'model_id': model_id,
        'cache_key': cache_key
    })

@app.route('/download_classified_data/<cache_key>')  # Add <cache_key> parameter
def download_classified_data(cache_key):
    if cache_key not in classified_data_cache:
        return "No classified data available for this file", 400
    
    predict_df = pd.DataFrame(classified_data_cache[cache_key])
    
    output = io.StringIO()
    predict_df.to_csv(output, index=False)
    output.seek(0)
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode())
    mem.seek(0)
    
    # Use a more descriptive filename
    download_name = f'classified_dataset_{cache_key}.csv'
    
    return send_file(
        mem,
        as_attachment=True,
        download_name=download_name,
        mimetype='text/csv'
    )

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum file size is 200MB'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    app.run(debug=True)



















