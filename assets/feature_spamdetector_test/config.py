# Configuration settings for Spam Slayer

# Model Configuration
MODEL_CONFIG = {
    "model_name": "intfloat/multilingual-e5-base",  # Embedding model
    "batch_size": 32,  # Batch size for embedding generation
    "max_length": 512,  # Maximum token length
    "test_size": 0.2,  # Train-test split ratio
    "random_state": 42,  # Random seed for reproducibility
}

# Dataset Configuration
DATASET_CONFIG = {
    "kaggle_dataset": "victorhoward2/vietnamese-spam-post-in-social-network",
    "kaggle_file": "vi_dataset.csv",
    "gdrive_file_id": "1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R",  # Default Google Drive file ID
    "augmentation": {
        "enabled": True,
        "aug_ratio": 0.2,  # Synonym replacement ratio
        "alpha": 0.3,  # Hard ham generation ratio
    }
}

# Classification Configuration
CLASSIFICATION_CONFIG = {
    "default_k": 5,  # Default number of neighbors
    "alpha_range": (0.0, 1.1, 0.1),  # Alpha optimization range (start, stop, step)
    "k_values": [1, 3, 5],  # K values for evaluation
    "confidence_threshold": 0.7,  # Minimum confidence for predictions
}

# UI Configuration
UI_CONFIG = {
    "page_title": "🛡️ Spam Slayer",
    "page_icon": "🛡️",
    "layout": "wide",
    "max_input_length": 5000,  # Maximum characters for input text
    "default_language": "Vietnamese",  # Default language selection
    "show_advanced_options": True,  # Show advanced options by default
}

# File Paths
FILE_PATHS = {
    "model_artifacts": "model_artifacts.pkl",
    "faiss_index": "faiss_index.bin",
    "train_metadata": "train_metadata.json",
    "class_weights": "class_weights.json",
    "model_config": "model_config.json",
    "results_file": "results.json",
}

# Spam Subcategory Configuration
SUBCATEGORY_CONFIG = {
    "categories": {
        "spam_quangcao": {
            "name": "📢 Promotional/Advertisement",
            "description": "Marketing, sales, and promotional messages",
            "color": "#ff6b6b"
        },
        "spam_hethong": {
            "name": "⚠️ System Alert/Phishing",
            "description": "Fake security alerts and phishing attempts",
            "color": "#ffa726"
        },
        "spam_khac": {
            "name": "🔍 Other Spam Type",
            "description": "Other types of spam messages",
            "color": "#ab47bc"
        }
    }
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "enable_gpu": True,  # Use GPU if available
    "max_training_time": 1800,  # Maximum training time in seconds (30 minutes)
    "memory_limit_gb": 4,  # Soft memory limit in GB
    "progress_update_interval": 0.1,  # Progress update frequency
}

# Debugging Configuration
DEBUG_CONFIG = {
    "enable_logging": True,
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "save_debug_info": False,  # Save debug information to files
    "verbose_training": True,  # Show detailed training progress
}

# Security Configuration
SECURITY_CONFIG = {
    "sanitize_input": True,  # Sanitize user input
    "max_file_size_mb": 100,  # Maximum file size for uploads
    "allowed_file_types": [".csv", ".txt", ".json"],  # Allowed file extensions
}

# Export all configurations
CONFIG = {
    "model": MODEL_CONFIG,
    "dataset": DATASET_CONFIG,
    "classification": CLASSIFICATION_CONFIG,
    "ui": UI_CONFIG,
    "files": FILE_PATHS,
    "subcategory": SUBCATEGORY_CONFIG,
    "performance": PERFORMANCE_CONFIG,
    "debug": DEBUG_CONFIG,
    "security": SECURITY_CONFIG,
}

# Helper functions to get configuration values
def get_config(section, key=None, default=None):
    """Get configuration value"""
    if section not in CONFIG:
        return default
    
    if key is None:
        return CONFIG[section]
    
    return CONFIG[section].get(key, default)

def update_config(section, key, value):
    """Update configuration value"""
    if section in CONFIG:
        CONFIG[section][key] = value

def get_model_config():
    """Get model configuration"""
    return MODEL_CONFIG

def get_dataset_config():
    """Get dataset configuration"""
    return DATASET_CONFIG

def get_ui_config():
    """Get UI configuration"""
    return UI_CONFIG