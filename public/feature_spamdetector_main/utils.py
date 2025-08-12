import os
import json
import pickle
import faiss
from datetime import datetime

def load_model_artifacts():
    """Load model artifacts from files"""
    model_files = [
        'model_artifacts.pkl',
        'faiss_index.bin',
        'train_metadata.json',
        'class_weights.json',
        'model_config.json'
    ]
    
    # Check if all files exist
    if not all(os.path.exists(f) for f in model_files):
        return None
    
    try:
        # Load artifacts
        with open("model_artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)
        
        with open("model_config.json", "r") as f:
            config = json.load(f)
        
        with open("train_metadata.json", "r", encoding="utf-8") as f:
            train_metadata = json.load(f)
        
        with open("class_weights.json", "r") as f:
            class_weights = json.load(f)
        
        index = faiss.read_index("faiss_index.bin")
        
        return {
            'artifacts': artifacts,
            'config': config,
            'train_metadata': train_metadata,
            'class_weights': class_weights,
            'index': index
        }
    
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None

def save_model_artifacts(artifacts_dict):
    """Save model artifacts to files"""
    try:
        # Save artifacts
        with open("model_artifacts.pkl", "wb") as f:
            pickle.dump(artifacts_dict['artifacts'], f)
        
        with open("model_config.json", "w") as f:
            json.dump(artifacts_dict['config'], f, indent=2)
        
        with open("train_metadata.json", "w", encoding="utf-8") as f:
            json.dump(artifacts_dict['train_metadata'], f, ensure_ascii=False, indent=2)
        
        with open("class_weights.json", "w") as f:
            json.dump(artifacts_dict['class_weights'], f, indent=2)
        
        faiss.write_index(artifacts_dict['index'], "faiss_index.bin")
        
        return True
    
    except Exception as e:
        print(f"Error saving model artifacts: {e}")
        return False

def check_model_files_exist():
    """Check if model files exist"""
    required_files = [
        'model_artifacts.pkl',
        'faiss_index.bin',
        'train_metadata.json',
        'class_weights.json',
        'model_config.json'
    ]
    
    existing_files = [f for f in required_files if os.path.exists(f)]
    missing_files = [f for f in required_files if f not in existing_files]
    
    return {
        'all_exist': len(missing_files) == 0,
        'existing': existing_files,
        'missing': missing_files
    }

def get_model_info():
    """Get information about the saved model"""
    if not os.path.exists("model_config.json"):
        return None
    
    try:
        with open("model_config.json", "r") as f:
            config = json.load(f)
        
        return config.get('model_info', {})
    
    except Exception as e:
        print(f"Error loading model info: {e}")
        return None

def cleanup_model_files():
    """Clean up model files"""
    files_to_remove = [
        'model_artifacts.pkl',
        'faiss_index.bin',
        'train_metadata.json',
        'class_weights.json',
        'model_config.json'
    ]
    
    removed_files = []
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                removed_files.append(file)
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    return removed_files

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def get_model_files_info():
    """Get information about model files"""
    files_info = {}
    
    model_files = [
        'model_artifacts.pkl',
        'faiss_index.bin',
        'train_metadata.json',
        'class_weights.json',
        'model_config.json'
    ]
    
    for file in model_files:
        if os.path.exists(file):
            stat = os.stat(file)
            files_info[file] = {
                'size': format_file_size(stat.st_size),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            files_info[file] = {'exists': False}
    
    return files_info

def validate_input_text(text):
    """Validate input text for classification"""
    if not text or not isinstance(text, str):
        return False, "Text must be a non-empty string"
    
    text = text.strip()
    if len(text) == 0:
        return False, "Text cannot be empty"
    
    if len(text) > 5000:
        return False, "Text is too long (max 5000 characters)"
    
    return True, text

def create_download_link(file_path, link_text):
    """Create a download link for files (for Streamlit)"""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    import base64
    b64 = base64.b64encode(data).decode()
    
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href