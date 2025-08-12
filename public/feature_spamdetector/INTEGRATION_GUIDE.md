# Integration Guide: Using run_enhanced_pipeline for Improved Training

## Issues Found and Required Fixes

### 1. Critical Issues in `spam_model_train.py`

#### Missing Global Variables and Dependencies
The `run_enhanced_pipeline` function uses several global variables that aren't properly defined:
- `model`, `tokenizer`, `device` - Not initialized properly
- `model_name` - Referenced but not defined globally
- Several function dependencies missing

#### Missing Imports
```python
# Add these imports at the top of spam_model_train.py
import pickle
import os
```

### 2. Key Functions That Need Fixes

#### A. Missing/Incomplete Function Bodies
Several functions referenced by `run_enhanced_pipeline` are incomplete:

1. **Global model initialization** - Missing at top level
2. **`load_dataset()` function** - Has placeholder
3. **Several helper functions** have empty implementations

### 3. Integration Steps for `spam_model.py`

#### Step 1: Create Enhanced Training Method
Add this method to your `SpamClassifier` class:

```python
def train_enhanced(self, messages, labels, test_size=0.2, use_augmentation=True,
                  aug_ratio=0.2, alpha_hard_ham=0.3, progress_callback=None):
    """Enhanced training with data augmentation"""

    if progress_callback:
        progress_callback(0.05, "Starting enhanced training...")

    # Import augmentation functions
    try:
        from spam_model_train import augment_dataset
    except ImportError:
        print("Warning: augmentation not available, proceeding without it")
        use_augmentation = False

    # Data augmentation
    original_count = len(messages)
    if use_augmentation:
        if progress_callback:
            progress_callback(0.10, "Augmenting dataset...")
        try:
            aug_messages, aug_labels = augment_dataset(
                messages, labels, aug_ratio=aug_ratio, alpha=alpha_hard_ham
            )
            if aug_messages:
                messages = messages + aug_messages
                labels = labels + aug_labels
                print(f"📈 Dataset augmented: {original_count} → {len(messages)}")
        except Exception as e:
            print(f"Augmentation failed: {e}, continuing without augmentation")

    # Continue with normal training
    return self.train(messages, labels, test_size=test_size, progress_callback=progress_callback)
```

#### Step 2: Create Standalone Enhanced Pipeline Function
Create a new file `enhanced_training.py`:

```python
import numpy as np
import torch
import faiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import json
from collections import Counter
from spam_model import SpamClassifier
from spam_model_train import augment_dataset

def run_enhanced_pipeline_standalone(messages, labels, classification_language='English',
                                   test_size=0.2, use_augmentation=True,
                                   aug_ratio=0.2, alpha_hard_ham=0.3):
    """
    Standalone enhanced pipeline that can be imported into spam_model.py
    """
    print("=== Enhanced Spam Classification Pipeline ===")

    # 1. Initialize classifier
    classifier = SpamClassifier(classification_language=classification_language)

    # 2. Data augmentation
    original_count = len(messages)
    if use_augmentation:
        print("\n=== Data Augmentation ===")
        try:
            aug_messages, aug_labels = augment_dataset(
                messages, labels, aug_ratio=aug_ratio, alpha=alpha_hard_ham
            )
            if aug_messages:
                messages = messages + aug_messages
                labels = labels + aug_labels
                print(f"📈 Dataset size: {original_count} → {len(messages)}")
        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}")

    # 3. Train with enhanced dataset
    results = classifier.train(messages, labels, test_size=test_size)

    # 4. Save enhanced model
    classifier.save_to_files()

    # 5. Return results
    return {
        'classifier': classifier,
        'original_dataset_size': original_count,
        'final_dataset_size': len(messages),
        'augmentation_count': len(messages) - original_count,
        'training_results': results
    }
```

### 4. Fixed Requirements for `spam_model_train.py`

Here are the critical fixes needed in the training file:

#### A. Add Missing Global Initialization
```python
# Add after imports, before function definitions
def initialize_global_model():
    """Initialize global model components"""
    global model, tokenizer, device, model_name

    token_path = Path("./tokens/hugging_face_token.txt")
    hf_token = None
    if token_path.exists():
        with open(token_path, 'r') as f:
            hf_token = f.read().strip()

    model_name = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModel.from_pretrained(model_name, token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, tokenizer, device

# Initialize immediately
model, tokenizer, device = initialize_global_model()
```

#### B. Fix Function Dependencies
The functions `get_embeddings`, `calculate_class_weights`, etc. need to be properly implemented.

### 5. Usage Examples

#### Option 1: Use Enhanced Training in app.py
```python
# In train_model_callback function in app.py
def train_model_callback(classification_language):
    try:
        classifier = SpamClassifier(classification_language=classification_language)

        # Load data
        data_source = 'kaggle' if classification_language == 'Vietnamese' else 'gdrive'
        messages, labels = classifier.load_dataset(source=data_source)

        # Use enhanced training instead of regular training
        results = classifier.train_enhanced(
            messages, labels,
            use_augmentation=True,
            aug_ratio=0.2,
            alpha_hard_ham=0.3,
            progress_callback=lambda p, msg: (
                progress_bar.progress(min(30 + int(p * 0.6), 90)),
                status_text.text(msg)
            )
        )

        # ... rest of the function
```

#### Option 2: Use Standalone Pipeline
```python
# Create train_enhanced.py script
from enhanced_training import run_enhanced_pipeline_standalone
from spam_model import SpamClassifier

def main():
    # Load data
    classifier = SpamClassifier(classification_language='English')
    messages, labels = classifier.load_dataset(source='gdrive')

    # Run enhanced pipeline
    results = run_enhanced_pipeline_standalone(
        messages, labels,
        classification_language='English',
        use_augmentation=True
    )

    print("Enhanced training completed!")
    print(f"Original dataset: {results['original_dataset_size']}")
    print(f"Final dataset: {results['final_dataset_size']}")
    print(f"Augmented samples: {results['augmentation_count']}")

if __name__ == "__main__":
    main()
```

### 6. Testing the Integration

1. **Fix spam_model_train.py first:**
   - Add missing imports and global variables
   - Implement placeholder functions

2. **Test augmentation separately:**
   ```python
   from spam_model_train import augment_dataset
   test_messages = ["Hello world", "Buy now!"]
   test_labels = ["ham", "spam"]
   aug_msg, aug_lab = augment_dataset(test_messages, test_labels)
   print(f"Generated {len(aug_msg)} augmented samples")
   ```

3. **Integrate step by step:**
   - Start with `train_enhanced()` method
   - Test with small dataset
   - Gradually add full pipeline

### 7. Expected Improvements

Using the enhanced pipeline should provide:
- **Balanced datasets** through hard ham generation
- **Increased diversity** through synonym replacement
- **Better generalization** on subtle spam detection
- **Improved accuracy** especially for social engineering attacks

### 8. Troubleshooting

**Common Issues:**
- `NameError: name 'model' is not defined` → Fix global model initialization
- Import errors → Check all required packages in requirements.txt
- Memory issues → Reduce batch size or dataset size for testing
- Augmentation not working → Check NLTK wordnet installation

**Quick Debug Commands:**
```python
# Test if augmentation works
python -c "from spam_model_train import augment_dataset; print('Augmentation OK')"

# Test model loading
python -c "from spam_model import SpamClassifier; c=SpamClassifier(); print('Model OK')"
```
