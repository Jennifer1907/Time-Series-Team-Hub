# 🏗️ KIẾN TRÚC TỔNG QUAN - MODULE 5: HEART DISEASE PREDICTION

## 📋 **MỤC LỤC**
1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Cấu Trúc Thư Mục](#2-cấu-trúc-thư-mục)
3. [Chi Tiết Từng Component](#3-chi-tiết-từng-component)
4. [Workflow Diagram](#4-workflow-diagram)
5. [Data Flow](#5-data-flow)

---

## **1. TỔNG QUAN HỆ THỐNG**

Hệ thống dự đoán bệnh tim sử dụng **Multimodal Fusion Model** kết hợp:
- 📊 **Numerical Data**: 70,000 bệnh nhân (Sulianova dataset)
- 🎥 **Video Data**: 10,000 video siêu âm tim (EchoNet-Dynamic)
- 🧠 **Deep Learning**: MLP + ResNet-50 CNN + Fusion Layer
- 🚀 **Production API**: FastAPI với video selection thông minh

---

## **2. CẤU TRÚC THỨ MỤC**

```
Module5/
├── 📓 heart_prediction.ipynb       # Notebook huấn luyện & testing
├── 📊 cleveland.csv                # Dataset y tế (backup)
├── 📁 fusion_results.json          # Kết quả 5-fold CV
├── 📄 requirements.txt             # Dependencies
│
├── 📂 api/                         # ⭐ PRODUCTION API
│   ├── main.py                     # FastAPI endpoints
│   ├── 📁 utility/                 # Core logic
│   │   ├── model.py                # Model architecture
│   │   ├── load_model.py           # Load weights & scalers
│   │   ├── load_data.py            # Load datasets
│   │   ├── preprocessing.py        # Preprocessing pipeline
│   │   ├── predict.py              # Prediction logic
│   │   └── feature_engineer.py     # Feature engineering
│   ├── 📁 test/                    # Unit tests
│   ├── start_api.bat/sh            # Startup scripts
│   └── 📚 Documentation/
│       ├── README.md
│       ├── API_USAGE_GUIDE.md
│       ├── VIDEO_SELECTION_GUIDE.md
│       └── CODE_DOCUMENTATION_SUMMARY.md
│
├── 📂 data/                        # ⭐ PREPROCESSED DATA
│   ├── echonet_features.npy        # CNN features (10000×64)
│   ├── echonet_labels.npy          # Ground truth labels
│   └── echonet_features2.npy       # Alternative features
│
├── 📂 models/                      # ⭐ TRAINED MODELS
│   ├── best_fusion_model.pth       # Best model weights
│   ├── fusion_model_fold_*.pth     # 5-fold models
│   ├── num_scaler_fold_*.pkl       # Numerical scalers
│   └── cnn_scaler_fold_*.pkl       # CNN feature scalers
│
└── 📂 ui/                          # Frontend (if exists)
```

---

## **3. CHI TIẾT TỪNG COMPONENT**

### **📂 A. API FOLDER** (`api/`) - **TRỌNG TÂM**

Đây là **production deployment** với FastAPI, cung cấp REST API cho dự đoán bệnh tim.

#### **🔧 A1. Core Files**

##### **`main.py`** - FastAPI Application
**Chức năng chính:**
- **Endpoints:**
  - `GET /` - API information
  - `GET /health` - Health check
  - `GET /videos` - List available videos (pagination)
  - `GET /videos/search` - Search videos by filename
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions

**Đặc điểm nổi bật:**
```python
# Flexible video selection
{
    "patient_data": {...},
    "video_index": 2           # Option 1: By index
    # OR
    "video_filename": "0X10...avi"  # Option 2: By filename
}
```

**Models:**
- `PatientData`: 13 fields (age, gender, height, weight, BP, cholesterol...)
  - Auto-convert age: years → days (50 years → 18,250 days)
- `PredictionRequest`: patient_data + video selection
- `PredictionResponse`: prediction + probability + confidence + metadata

---

#### **🛠️ A2. Utility Package** (`api/utility/`)

##### **`model.py`** - Model Architecture
```python
class NumericalMLP(nn.Module):
    """MLP cho 28 numerical features"""
    Input: (batch, 28) → [128, 64] → Output: (batch, 64)

class FeatureFusionModel(nn.Module):
    """Multimodal fusion"""
    Inputs:
        - Numerical: (batch, 28) → MLP → (batch, 64)
        - CNN: (batch, 64) → Identity
    Fusion: Concat(64, 64) → [128, 64] → Classifier → (batch, 2)
```

**Kiến trúc:**
```
Numerical Features (28) ──→ MLP ──→ (64)
                                      ├──→ Fusion Layer ──→ Classifier ──→ (2)
CNN Features (64) ────────────────→ (64)
```

##### **`load_model.py`** - Model & Scaler Loading
```python
# Load trained model
fusion_model = FeatureFusionModel(numerical_dim=28, cnn_dim=64)
fusion_model.load_state_dict(torch.load('best_fusion_model.pth'))

# Load scalers
num_scaler = pickle.load('num_scaler_fold_1.pkl')    # StandardScaler (28 features)
cnn_scaler = pickle.load('cnn_scaler_fold_1.pkl')    # StandardScaler (64 features)

# Load fixed CNN model (ResNet-50)
cnn_model = FixedCNNFeatureExtractor()
```

##### **`load_data.py`** - Dataset Loading
```python
# Load Sulianova dataset (70,000 patients)
sulianova_data = kagglehub.dataset_load(
    "sulianova/cardiovascular-disease-dataset",
    file_path="cardio_train.csv"
)

# Load EchoNet-Dynamic (10,000 videos)
def load_echonet_with_kagglehub():
    """Returns: (FileList, VolumeTracings, Videos_path)"""
    # FileList.csv: 10,030 videos with metadata
    # Videos/: .avi video files
```

##### **`feature_engineer.py`** - Feature Engineering
```python
def preprocess_and_engineer_features(df):
    """
    28 features from 13 raw features:
    - Basic: age, gender, height, weight, BP, cholesterol, gluc
    - Derived: BMI, MAP, PP, cholesterol_ratio, etc.
    - Interactions: bmi_x_age, map_x_age, pp_x_age, etc.
    """
    # Age in years
    df['age_years'] = df['age'] / 365

    # BMI
    df['bmi'] = df['weight'] / ((df['height']/100)**2)

    # MAP (Mean Arterial Pressure)
    df['map'] = (df['ap_hi'] + 2*df['ap_lo']) / 3

    # ... + 20 more features
    return X (28 features), y (labels)
```

##### **`preprocessing.py`** - Preprocessing Pipeline
```python
def preprocessing_pipeline(sample_json, video_index, ...):
    """
    STEP 1: Numerical Data
        1. Convert JSON → DataFrame
        2. Feature engineering (13 → 28 features)
        3. Scale with num_scaler
        4. Convert to tensor

    STEP 2: CNN Features
        Option A: Load pre-extracted features
            - echonet_features.npy[video_index]
        Option B: Extract on-the-fly
            - Read video file
            - Extract frames
            - CNN inference → 64 features
        5. Scale with cnn_scaler
        6. Convert to tensor

    STEP 3: Prepare for model
        - Move to device (CPU/CUDA)
        - Return: (numerical_tensor, cnn_tensor, metadata)
    """
```

##### **`predict.py`** - Prediction Logic
```python
def predict_single_sample(sample_json, video_index, ...):
    """
    1. Preprocess data (numerical + video)
    2. Model inference
    3. Softmax → probabilities
    4. Return: (prediction, probability, info)
    """
    # Preprocess
    numerical_input, cnn_input, info = preprocessing_pipeline(...)

    # Predict
    fusion_model.eval()
    with torch.no_grad():
        outputs, _ = fusion_model(numerical_input, cnn_input)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = outputs.argmax(dim=1).item()
        disease_prob = probabilities[0, 1].item()

    return prediction, disease_prob, info
```

---

#### **📚 A3. Documentation Files**

| File | Nội Dung |
|------|----------|
| `README.md` | Hướng dẫn setup & deployment |
| `API_USAGE_GUIDE.md` | Cách sử dụng API, examples với requests/curl |
| `VIDEO_SELECTION_GUIDE.md` | Hướng dẫn chọn video (index vs filename) |
| `CODE_DOCUMENTATION_SUMMARY.md` | Giải thích code comments đã thêm |
| `QUICKSTART.txt` | Quick start guide |

---

### **📂 B. DATA FOLDER** (`data/`)

Chứa **pre-extracted CNN features** để tăng tốc inference.

```python
# echonet_features.npy
Shape: (10000, 64)
Content: ResNet-50 features cho 10,000 videos
Size: ~5MB (thay vì 7.3GB videos)

# echonet_labels.npy
Shape: (10000,)
Content: Ground truth labels (ejection fraction, etc.)
```

**Lợi ích:**
- ✅ **Fast inference**: Không cần chạy CNN mỗi lần predict
- ✅ **Memory efficient**: 5MB vs 7.3GB
- ✅ **Reproducible**: Cùng features cho mọi prediction

**Cách tạo:**
```python
# Trong notebook (cell trích xuất features)
cnn_features = []
for video_path in video_paths:
    features = cnn_model.extract_features(video_path)
    cnn_features.append(features)

np.save('echonet_features.npy', np.array(cnn_features))
```

---

### **📂 C. MODELS FOLDER** (`models/`)

Chứa **trained model weights** và **scalers** từ 5-fold cross-validation.

```
models/
├── best_fusion_model.pth          # Model tốt nhất (fold với highest accuracy)
│
├── fusion_model_fold_1.pth        # Fold 1 weights
├── fusion_model_fold_2.pth        # Fold 2 weights
├── fusion_model_fold_3.pth        # Fold 3 weights
├── fusion_model_fold_4.pth        # Fold 4 weights
├── fusion_model_fold_5.pth        # Fold 5 weights
│
├── num_scaler_fold_1.pkl          # Numerical scaler (StandardScaler)
├── num_scaler_fold_2.pkl          #   - mean: (28,)
├── num_scaler_fold_3.pkl          #   - std: (28,)
├── num_scaler_fold_4.pkl
├── num_scaler_fold_5.pkl
│
├── cnn_scaler_fold_1.pkl          # CNN feature scaler (StandardScaler)
├── cnn_scaler_fold_2.pkl          #   - mean: (64,)
├── cnn_scaler_fold_3.pkl          #   - std: (64,)
├── cnn_scaler_fold_4.pkl
└── cnn_scaler_fold_5.pkl
```

**File sizes:**
- `.pth` files: ~1-5MB (model weights)
- `.pkl` files: ~1-5KB (scaler parameters)

**Nội dung state_dict:**
```python
torch.load('best_fusion_model.pth'):
{
    'numerical_mlp.mlp.0.weight': tensor(128, 28),
    'numerical_mlp.mlp.0.bias': tensor(128),
    'numerical_mlp.mlp.3.weight': tensor(64, 128),
    'numerical_mlp.mlp.3.bias': tensor(64),
    'fusion_layer.0.weight': tensor(128, 128),
    'fusion_layer.0.bias': tensor(128),
    'fusion_layer.3.weight': tensor(64, 128),
    'fusion_layer.3.bias': tensor(64),
    'classifier.0.weight': tensor(32, 64),
    'classifier.0.bias': tensor(32),
    'classifier.3.weight': tensor(2, 32),
    'classifier.3.bias': tensor(2)
}
```

---

### **📓 D. NOTEBOOK** (`heart_prediction.ipynb`)

**Mục đích:**
- 🧪 **Experimentation & Training**
- 📊 **Model Evaluation**
- 🧪 **Testing & Validation**

**Nội dung chính:**

#### **Section 1: Data Loading & EDA**
```python
# Load Sulianova dataset
sulianova_data = load_sulianova()  # 70,000 patients

# Load EchoNet dataset
echonet_filelist, videos_path = load_echonet()  # 10,000 videos

# EDA
print(sulianova_data.describe())
visualize_distributions()
check_missing_values()
```

#### **Section 2: Feature Engineering**
```python
# Create 28 features from 13 raw features
X, y, feature_names = preprocess_and_engineer_features(sulianova_data)

# Features:
# - Basic: age_years, gender, height, weight, ...
# - Derived: bmi, map, pp, pulse_pressure_ratio, ...
# - Interactions: bmi_x_age, map_x_age, ...
```

#### **Section 3: CNN Feature Extraction**
```python
# Extract ResNet-50 features from videos
cnn_model = FixedCNNFeatureExtractor()

echonet_features = []
for video_path in video_paths:
    features = extract_video_features(video_path, cnn_model)
    echonet_features.append(features)

# Save for reuse
np.save('data/echonet_features.npy', echonet_features)
```

#### **Section 4: Model Training (5-Fold CV)**
```python
# 5-fold cross-validation
fusion_results = train_fusion_model_with_cv(
    numerical_X=X,              # (70000, 28)
    numerical_y=y,              # (70000,)
    cnn_features=echonet_features,  # (10000, 64)
    n_folds=5,
    epochs=50,
    batch_size=64
)

# Save each fold
for fold_result in fusion_results['fold_results']:
    fold_num = fold_result['fold']
    model = fold_result['model']

    # Save model
    torch.save(model.state_dict(),
               f'models/fusion_model_fold_{fold_num}.pth')

    # Save scalers
    pickle.dump(fold_result['num_scaler'],
                f'models/num_scaler_fold_{fold_num}.pkl')
    pickle.dump(fold_result['cnn_scaler'],
                f'models/cnn_scaler_fold_{fold_num}.pkl')
```

#### **Section 5: Model Evaluation**
```python
# Metrics per fold
for fold_result in fusion_results['fold_results']:
    print(f"Fold {fold_result['fold']}:")
    print(f"  Train Acc: {fold_result['train_accuracy']:.4f}")
    print(f"  Val Acc: {fold_result['val_accuracy']:.4f}")
    print(f"  Val Loss: {fold_result['val_loss']:.4f}")

# Select best model
best_fold = max(fusion_results['fold_results'],
                key=lambda x: x['val_accuracy'])
```

#### **Section 6: Testing & Inference**
```python
# Test single prediction
sample_patient = {
    "id": 12345,
    "age": 50,
    "gender": 2,
    "height": 168,
    "weight": 62.0,
    "ap_hi": 110,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
}

prediction, probability, info = predict_single_sample(
    sample_json=sample_patient,
    video_index=0,
    fusion_model=fusion_model,
    num_scaler=num_scaler,
    cnn_scaler=cnn_scaler
)

print(f"Prediction: {prediction}")
print(f"Disease Probability: {probability:.2%}")
```

---

## **4. WORKFLOW DIAGRAM**

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           TRAINING WORKFLOW                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA LOADING                                                       │
└─────────────────────────────────────────────────────────────────────────────┘

    [Sulianova Dataset]              [EchoNet Dataset]
     (70,000 patients)                (10,000 videos)
           │                                  │
           │                                  │
           ▼                                  ▼
    ┌─────────────┐                  ┌────────────────┐
    │ Load CSV    │                  │ Load Videos    │
    │ via         │                  │ via            │
    │ KaggleHub   │                  │ KaggleHub      │
    └──────┬──────┘                  └────────┬───────┘
           │                                  │
           │                                  │
           ▼                                  ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: FEATURE ENGINEERING                                                │
└─────────────────────────────────────────────────────────────────────────────┘

    [Raw Features: 13]               [Video Files (.avi)]
    ┌──────────────────┐                    │
    │ age, gender,     │                    │
    │ height, weight,  │                    ▼
    │ ap_hi, ap_lo,    │          ┌──────────────────────┐
    │ cholesterol,     │          │ ResNet-50 CNN        │
    │ gluc, smoke,     │          │ Feature Extractor    │
    │ alco, active     │          │ (Pre-trained)        │
    └────────┬─────────┘          └──────────┬───────────┘
  (Raw features combine                      │
   with engineered feature)                  │
             ▼                               ▼
    ┌─────────────────────┐        ┌─────────────────────┐
    │ Feature Engineering │        │ Extract Features    │
    │ • BMI, MAP, PP      │        │ • 10 frames/video   │
    │ • Ratios            │        │ • Average pooling   │
    │ • Interactions      │        │ • Output: (10k, 64) │
    └────────┬────────────┘        └─────────┬───────────┘
			│                                │
			▼                                ▼
    [Engineered: 28 features]     [CNN Features: 64 dims]
             │                               │
             │                               ▼
             │                    ┌──────────────────────┐
             │                    │ Save Features        │
             │                    │ echonet_features.npy │
             │                    └──────────────────────┘
             │
             ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: PREPROCESSING & SCALING                                             │
└─────────────────────────────────────────────────────────────────────────────┘

    [Numerical: 28]               [CNN Features: 64]
           │                              │
           ▼                              ▼
    ┌─────────────────┐          ┌─────────────────┐
    │ StandardScaler  │          │ StandardScaler  │
    │ Fit on Train    │          │ Fit on Train    │
    └────────┬────────┘          └────────┬────────┘
             │                            │
             ▼                            ▼
    [Scaled Numerical]            [Scaled CNN]
             │                            │
             │                            │
             └────────────┬───────────────┘
                          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: MODEL TRAINING (5-Fold CV)                                          │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌────────────────────────┐
                    │   Fusion Model         │
                    │  ┌──────────────────┐  │
    [Numerical] ──→ │  │ Numerical MLP    │  │
       (28)         │  │ [128] → [64]     │  │
                    │  └────────┬─────────┘  │
                    │           │            │
                    │           ▼            │
                    │  ┌──────────────────┐  │
                    │  │ Fusion Layer     │  │
                    │  │ Concat(64+64)    │  │
    [CNN] ─────────→│  │ → [128] → [64]   │  │
       (64)         │  └────────┬─────────┘  │
                    │           │            │
                    │           ▼            │
                    │  ┌──────────────────┐  │
                    │  │ Classifier       │  │
                    │  │ [64] → [32] → 2  │  │
                    │  └────────┬─────────┘  │
                    │           │            │
                    └───────────┼─────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Output: (Disease, 0/1)│
                    └───────────────────────┘

                          Training Loop:
                    ┌──────────────────────┐
                    │ Fold 1: Train & Val  │
                    ├──────────────────────┤
                    │ Fold 2: Train & Val  │
                    ├──────────────────────┤
                    │ Fold 3: Train & Val  │
                    ├──────────────────────┤
                    │ Fold 4: Train & Val  │
                    ├──────────────────────┤
                    │ Fold 5: Train & Val  │
                    └──────────┬───────────┘
                               ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: MODEL SAVING                                                       │
└─────────────────────────────────────────────────────────────────────────────┘

                    For Each Fold (1-5):
                    ┌────────────────────────┐
                    │ Save Model Weights     │
                    │ fusion_model_fold_k.pth│
                    └────────────────────────┘
                    ┌────────────────────────┐
                    │ Save Num Scaler        │
                    │ num_scaler_fold_k.pkl  │
                    └────────────────────────┘
                    ┌────────────────────────┐
                    │ Save CNN Scaler        │
                    │ cnn_scaler_fold_k.pkl  │
                    └────────────────────────┘

                    Select Best Fold:
                    ┌────────────────────────┐
                    │ Copy to:               │
                    │ best_fusion_model.pth  │
                    └────────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════════╗
║                          INFERENCE WORKFLOW (API)                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: API REQUEST                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

    Client (Browser/Postman/Code)
           │
           │ POST /predict
           │ {
           │   "patient_data": {
           │     "age": 50,
           │     "gender": 2,
           │     "height": 168,
           │     ...
           │   },
           │   "video_filename": "0X1005D03EED19C65B.avi"
           │ }
           ▼
    ┌────────────────┐
    │ FastAPI Server │
    │ (main.py)      │
    └────────┬───────┘
             │
             ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: DATA VALIDATION                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │ Pydantic Validation  │
    │ • Age: years→days    │
    │ • BP: valid range    │
    │ • Video: filename→idx│
    └──────────┬───────────┘
               ▼
    ┌──────────────────────┐
    │ Load Models/Scalers  │
    │ • fusion_model.pth   │
    │ • num_scaler.pkl     │
    │ • cnn_scaler.pkl     │
    └──────────┬───────────┘
               ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: PREPROCESSING                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

    [Patient Data]              [Video Index]
         │                            │
         ▼                            ▼
    ┌─────────────────┐      ┌──────────────────┐
    │ Feature Eng.    │      │ Load CNN Features│
    │ 13 → 28 features│      │ from .npy file   │
    └────────┬────────┘      └────────┬─────────┘
             │                        │
             ▼                        ▼
    ┌─────────────────┐      ┌──────────────────┐
    │ num_scaler      │      │ cnn_scaler       │
    │ .transform()    │      │ .transform()     │
    └────────┬────────┘      └────────┬─────────┘
             │                        │
             ▼                        ▼
    [Scaled Numerical (28)]   [Scaled CNN (64)]
             │                        │
             └────────────┬───────────┘
                          ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: MODEL INFERENCE                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌────────────────────┐
                    │  Fusion Model      │
                    │  .eval() mode      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Forward Pass       │
                    │ (no gradient)      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Softmax            │
                    │ → Probabilities    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Argmax             │
                    │ → Prediction (0/1) │
                    └─────────┬──────────┘
                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: RESPONSE                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌────────────────────┐
                    │ Build Response     │
                    │ {                  │
                    │   prediction: 0,   │
                    │   label: "No Dis", │
                    │   probability: 0.12│
                    │   confidence: 0.88 │
                    │ }                  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ JSON Response      │
                    └─────────┬──────────┘
                              │
                              ▼
                         Client Receives
```

---

## **5. DATA FLOW**

### **📊 Training Data Flow**

```
┌──────────────────────────────────────────────────────────────────┐
│ Sulianova Dataset (70,000 patients)                              │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Raw: id, age, gender, height, weight, ap_hi, ap_lo,          │ │
│ │      cholesterol, gluc, smoke, alco, active, cardio          │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Feature Engineering: 13 → 28 features                        │ │
│ │ • age_years, bmi, map, pp, pulse_pressure_ratio              │ │
│ │ • cholesterol_ratio, gluc_ratio, bp_age_interaction          │ │
│ │ • bmi_x_age, map_x_age, pp_x_age, etc.                       │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ StandardScaler Fit                                           │ │
│ │ X_scaled = (X - mean) / std                                  │ │
│ └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ EchoNet Dataset (10,000 videos)                                  │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Videos: *.avi files (7.3GB total)                            │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ ResNet-50 Feature Extraction                                 │ │
│ │ • Sample 10 frames per video                                 │ │
│ │ • Extract features: (10, 64)                                 │ │
│ │ • Average pooling: (64,)                                     │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Save Features: echonet_features.npy (10000, 64)              │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ StandardScaler Fit                                           │ │
│ │ CNN_scaled = (CNN - mean) / std                              │ │
│ └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

         ┌──────────────────────────┬─────────────────────────┐
         │                          │                         │
         ▼                          ▼                         ▼
[Numerical Scaled (28)]   [CNN Scaled (64)]         [Labels (70k)]
         │                          │                         │
         └──────────────┬───────────┴─────────────────────────┘
                        ▼
              ┌──────────────────────┐
              │ 5-Fold Cross-Val     │
              │ Train Fusion Model   │
              └──────────────────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │ Save:                │
              │ • Model weights (.pth)
              │ • Scalers (.pkl)     │
              └──────────────────────┘
```

### **🚀 Inference Data Flow**

```
User Input                                    API Response
    │                                              ▲
    │ POST /predict                                │
    ▼                                              │
┌────────────────────┐                             │
│ Patient Data       │                             │
│ {                  │                             │
│   age: 50,         │                             │
│   gender: 2,       │                             │
│   height: 168,     │                             │
│   ...              │                             │
│   video: "0X10..." │                             │
│ }                  │                             │
└─────────┬──────────┘                             │
          │                                        │
          ▼                                        │
┌────────────────────┐                             │
│ Validation         │                             │
│ • Pydantic models  │                             │
│ • Age: years→days  │                             │
│ • Video: name→idx  │                             │
└─────────┬──────────┘                             │
          │                                        │
          ├───────────────┬────────────────┐       │
          ▼               ▼                ▼       │
┌──────────────┐  ┌──────────────┐  ┌──────────┐   │
│ Feature Eng. │  │ Load CNN     │  │ Load     │   │
│ 13 → 28      │  │ Features     │  │ Models   │   │
└──────┬───────┘  └──────┬───────┘  └────┬─────┘   │
       │                 │                │        │
       ▼                 ▼                ▼        │
┌──────────────┐  ┌──────────────┐  ┌──────────┐   │
│ num_scaler   │  │ cnn_scaler   │  │ fusion   │   │
│ .transform() │  │ .transform() │  │ model    │   │
└──────┬───────┘  └──────┬───────┘  └────┬─────┘   │
       │                 │                │        │
       └────────┬────────┴────────────────┘        │
                ▼                                  │
       ┌─────────────────┐                         │
       │ Model Inference │                         │
       │ • Forward pass  │                         │
       │ • Softmax       │                         │
       │ • Argmax        │                         │
       └────────┬────────┘                         │
                │                                  │
                ▼                                  │
       ┌─────────────────┐                         │
       │ Response        │                         │
       │ {               │─────────────────────────┘
       │   prediction: 0 │
       │   label: "..."  │
       │   prob: 0.12    │
       │   conf: 0.88    │
       │ }               │
       └─────────────────┘
```

---

## **6. TỔNG KẾT**

### **🎯 Điểm Mạnh Của Kiến Trúc**

1. **Modular Design**
   - ✅ Tách biệt training (notebook) và production (API)
   - ✅ Utility package có thể tái sử dụng
   - ✅ Dễ maintain và extend

2. **Performance Optimization**
   - ✅ Pre-extract CNN features → Fast inference
   - ✅ Cached scalers → Consistent preprocessing
   - ✅ 5-fold CV → Robust model selection

3. **Production Ready**
   - ✅ FastAPI với auto-documentation (Swagger)
   - ✅ Flexible video selection (index/filename)
   - ✅ Comprehensive error handling
   - ✅ Health check endpoint

4. **Scalability**
   - ✅ Batch prediction support
   - ✅ Pagination for video listing
   - ✅ Search functionality
   - ✅ Easy to add new features

### **📌 Workflow Summary**

```
TRAINING:
Notebook → Feature Engineering → Model Training → Save Weights/Scalers

DEPLOYMENT:
API Startup → Load Models → Wait for Requests

INFERENCE:
Request → Validate → Preprocess → Predict → Response
```

### **🔐 Key Files to Remember**

| Component | Files | Purpose |
|-----------|-------|---------|
| **Model Architecture** | `api/utility/model.py` | Define NumericalMLP & FeatureFusionModel |
| **Training** | `heart_prediction.ipynb` | Train, evaluate, save models |
| **Weights** | `models/*.pth` | Trained model parameters |
| **Scalers** | `models/*.pkl` | Preprocessing statistics |
| **Features** | `data/*.npy` | Pre-extracted CNN features |
| **API** | `api/main.py` | Production endpoints |
| **Preprocessing** | `api/utility/preprocessing.py` | Data pipeline |
| **Prediction** | `api/utility/predict.py` | Inference logic |

---

**🎉 Hệ thống hoàn chỉnh từ nghiên cứu (notebook) đến production (API)!**
