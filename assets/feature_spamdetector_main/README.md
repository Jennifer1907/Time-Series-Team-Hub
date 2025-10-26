# 🛡️ Spam Slayer - Advanced Spam Detection System

A sophisticated AI-powered spam detection system built with Streamlit, featuring machine learning models for Vietnamese text classification with real-time analysis and interactive visualizations.

## 📋 Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage Guide](#usage-guide)
6. [Model Training](#model-training)
7. [Prediction & Analysis](#prediction--analysis)
8. [Batch Processing](#batch-processing)
9. [Performance Metrics](#performance-metrics)
10. [Troubleshooting](#troubleshooting)
11. [Technical Architecture](#technical-architecture)

## ✨ Features

### Core Functionality
- **Real-time Spam Detection**: Instant classification of Vietnamese text messages
- **Advanced ML Models**: Uses multilingual E5 embeddings with FAISS similarity search
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Batch Processing**: Upload CSV files for bulk spam detection
- **Visual Analytics**: Real-time performance charts and confidence visualizations
- **Saliency Analysis**: Highlights important words contributing to spam classification

### Spam Categories
- 📢 **Promotional/Advertisement**: Marketing and sales messages
- ⚠️ **System Alert/Phishing**: Fake security alerts and phishing attempts
- 🔍 **Other Spam Types**: Various other spam categories
- ✅ **Ham (Not Spam)**: Legitimate messages

### Advanced Features
- **Data Augmentation**: Automatic text augmentation for better training
- **Model Optimization**: Hyperparameter tuning for optimal performance
- **Multi-language Support**: Primarily Vietnamese with multilingual capabilities
- **Export Results**: Download predictions and analysis reports

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 2GB free space
- **GPU**: Optional (CUDA-compatible for faster training)

### Required Accounts
- **Hugging Face Account**: For accessing transformer models
  - Create account at [huggingface.co](https://huggingface.co/)
  - Generate access token: Profile → Settings → Access Tokens
- **Kaggle Account**: For dataset access (optional)
  - Create account at [kaggle.com](https://www.kaggle.com/)
  - Download API credentials from Account → API

## 🚀 Installation

### Step 1: Environment Setup

```bash
# Clone or navigate to the project directory
cd path/to/feature_spamdetector

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `streamlit` - Web application framework
- `torch` - PyTorch for ML models
- `transformers` - Hugging Face transformer models
- `faiss-cpu` - Similarity search library
- `scikit-learn` - Machine learning utilities
- `pandas`, `numpy` - Data manipulation
- `nltk` - Natural language processing
- `plotly` - Interactive visualizations
- `gdown`, `kagglehub` - Dataset downloading

### Step 3: Configure Credentials

Create necessary credential files:

```bash
# Create Hugging Face token file (if needed)
echo "your_huggingface_token_here" > token.txt

# Configure Kaggle credentials (optional)
# Place kaggle.json in the project directory
```

## ⚙️ Configuration

### Basic Configuration

The system uses `config.py` for settings. Key configurations:

```python
# Model settings
MODEL_CONFIG = {
    "model_name": "intfloat/multilingual-e5-base",
    "batch_size": 32,
    "max_length": 512,
    "test_size": 0.2
}

# Classification settings
CLASSIFICATION_CONFIG = {
    "default_k": 5,  # Number of neighbors
    "confidence_threshold": 0.7
}
```

### Advanced Settings

Modify `config.py` to adjust:
- **Model parameters**: Batch size, sequence length
- **Performance settings**: GPU usage, memory limits
- **UI customization**: Themes, layout options
- **Security options**: Input validation, file size limits

## 📖 Usage Guide

### Starting the Application

```bash
# Launch the Streamlit app
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Main Interface Components

1. **Sidebar Navigation**
   - Model training options
   - Dataset selection
   - Configuration settings

2. **Main Dashboard**
   - Text input for real-time prediction
   - Batch file upload
   - Results visualization
   - Performance metrics

3. **Advanced Options**
   - Model fine-tuning
   - Custom dataset upload
   - Export functionality

## 🎯 Model Training

### Training Process

1. **Navigate to Training Section**
   - Select "Train New Model" from sidebar
   - Choose data source (Kaggle or Google Drive)

2. **Dataset Selection**
   ```
   📊 Available Datasets:
   - Vietnamese Spam Dataset (Kaggle)
   - Custom Dataset (Upload CSV)
   - Google Drive Dataset (File ID)
   ```

3. **Training Configuration**
   - **Augmentation**: Enable text augmentation for better performance
   - **Train-Test Split**: Default 80/20 split
   - **K-Values**: Number of neighbors for classification

4. **Start Training**
   ```
   🔄 Training Progress:
   ├── Loading dataset...
   ├── Preprocessing text...
   ├── Generating embeddings...
   ├── Building FAISS index...
   ├── Optimizing parameters...
   └── ✅ Training complete!
   ```

### Training Monitoring

The interface shows real-time progress:
- **Progress Bar**: Overall completion percentage
- **Current Step**: Detailed status of current operation
- **Estimated Time**: Remaining time for completion
- **Memory Usage**: System resource utilization

### Model Artifacts

After training, the following files are saved:
- `model_artifacts.pkl` - Core model components
- `faiss_index.bin` - Similarity search index
- `train_metadata.json` - Training metadata
- `class_weights.json` - Class balancing weights
- `model_config.json` - Model configuration

## 🔍 Prediction & Analysis

### Single Text Prediction

1. **Input Text**
   - Enter text in the main input box
   - Maximum 5,000 characters
   - Supports Vietnamese and English

2. **Get Prediction**
   - Click "Analyze Text" button
   - View real-time classification result
   - See confidence scores

3. **Result Interpretation**
   ```
   🛡️ Prediction Results:

   Classification: SPAM 🚨
   Confidence: 87.3%
   Category: Promotional/Advertisement

   📊 Confidence Breakdown:
   ├── Spam: 87.3%
   └── Ham: 12.7%
   ```

### Saliency Analysis

The system highlights important words:
- **Red highlighting**: Words contributing to spam classification
- **Intensity**: Darker red indicates higher importance
- **Interactive**: Hover for detailed scores

### Confidence Levels

- **High Confidence** (>80%): Very reliable prediction
- **Medium Confidence** (60-80%): Generally reliable
- **Low Confidence** (<60%): Manual review recommended

## 📁 Batch Processing

### CSV File Upload

1. **Prepare CSV File**
   ```csv
   message,label
   "Xin chào, bạn có muốn mua sản phẩm...",spam
   "Họp team lúc 2pm hôm nay",ham
   ```

2. **Upload Process**
   - Navigate to "Batch Analysis" tab
   - Upload CSV file (max 100MB)
   - Select text column
   - Choose label column (optional)

3. **Processing Results**
   - Real-time progress tracking
   - Downloadable results
   - Performance metrics

### Batch Results Format

```csv
original_text,predicted_label,confidence,predicted_category
"Text message here",spam,0.87,promotional
"Another message",ham,0.94,legitimate
```

## 📊 Performance Metrics

### Model Evaluation

The system provides comprehensive metrics:

1. **Accuracy Metrics**
   - Overall accuracy
   - Per-class precision/recall
   - F1-scores

2. **Confusion Matrix**
   - Visual representation of predictions
   - True positive/negative rates
   - Classification errors

3. **ROC Curves**
   - Area under curve (AUC)
   - Threshold optimization
   - Performance visualization

### Real-time Analytics

- **Prediction Distribution**: Charts showing spam vs ham ratios
- **Confidence Histograms**: Distribution of prediction confidence
- **Processing Speed**: Messages processed per second
- **Model Performance**: Accuracy trends over time

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: Package installation fails
```bash
# Solution: Upgrade pip and try again
python -m pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Issue**: CUDA/GPU not detected
```bash
# Check PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Runtime Errors

**Issue**: Memory errors during training
- Reduce batch size in `config.py`
- Close other applications
- Use CPU-only mode if necessary

**Issue**: Model loading fails
- Check if model files exist
- Verify file permissions
- Re-run training if files corrupted

**Issue**: Slow performance
- Enable GPU acceleration
- Reduce input text length
- Optimize batch size

#### Data Issues

**Issue**: CSV upload fails
- Check file format (UTF-8 encoding)
- Verify column names
- Ensure file size < 100MB

**Issue**: Poor prediction accuracy
- Retrain with more diverse data
- Enable data augmentation
- Adjust classification threshold

### Debug Mode

Enable detailed logging:
```python
# In config.py
DEBUG_CONFIG = {
    "enable_logging": True,
    "log_level": "DEBUG",
    "verbose_training": True
}
```

### Getting Help

1. **Check Error Messages**: Look for specific error details
2. **Review Logs**: Enable debug mode for detailed information
3. **Verify Setup**: Ensure all dependencies are installed
4. **Test with Sample Data**: Use provided test cases

## 🏗️ Technical Architecture

### System Components

```
📦 Spam Slayer Architecture
├── 🎨 Frontend (Streamlit)
│   ├── User Interface
│   ├── Interactive Visualizations
│   └── Real-time Updates
├── 🧠 ML Pipeline
│   ├── Text Preprocessing
│   ├── Embedding Generation (E5)
│   ├── FAISS Similarity Search
│   └── Classification Logic
├── 💾 Data Layer
│   ├── Model Artifacts
│   ├── Training Data
│   └── Configuration Files
└── 🔧 Utilities
    ├── Data Augmentation
    ├── Performance Monitoring
    └── Export Functions
```

### Key Technologies

- **Streamlit**: Web application framework
- **Transformers**: Multilingual E5 embeddings
- **FAISS**: Efficient similarity search
- **PyTorch**: Deep learning framework
- **Plotly**: Interactive visualizations
- **Scikit-learn**: ML utilities and metrics

### Model Pipeline

1. **Text Preprocessing**
   - Cleaning and normalization
   - Tokenization
   - Encoding preparation

2. **Embedding Generation**
   - Multilingual E5 model
   - 768-dimensional vectors
   - Batch processing optimization

3. **Similarity Search**
   - FAISS index construction
   - K-nearest neighbors
   - Distance-based classification

4. **Post-processing**
   - Confidence calculation
   - Category assignment
   - Result formatting

### Security Features

- **Input Sanitization**: Prevents malicious input
- **File Validation**: Secure file upload handling
- **Memory Management**: Prevents resource exhaustion
- **Error Handling**: Graceful error recovery

## 📝 Additional Resources

### Sample Data Format

Example training data structure:
```csv
text,label,category
"Khuyến mãi đặc biệt chỉ hôm nay!",spam,promotional
"Cuộc họp sẽ bắt đầu lúc 10h",ham,legitimate
"CẢNH BÁO: Tài khoản của bạn sẽ bị khóa",spam,phishing
```

### API Integration

For programmatic usage:
```python
from spam_model import SpamClassifier

# Initialize classifier
classifier = SpamClassifier()

# Load trained model
classifier.load_model()

# Make prediction
result = classifier.predict("Your text here")
print(f"Classification: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Performance Optimization

Tips for better performance:
- Use GPU when available
- Batch process multiple texts
- Cache embeddings for repeated texts
- Optimize memory usage with streaming

---

**Need help?** Check the troubleshooting section or review the error logs for detailed information.
