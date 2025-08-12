import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import pickle
import os
from datetime import datetime
from collections import Counter
import re
import warnings
import gdown
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Import augmentation functions from enhanced_training_clean
from enhanced_training_clean import enhanced_augmentation

warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj



class SpamClassifier:
    def __init__(self, model_name="intfloat/multilingual-e5-base", classification_language='English', aug_ratio=0.5, alpha_hard_ham=0.5):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_folder_path = f"model_resources/{classification_language}"
        self.aug_ratio = aug_ratio
        self.alpha_hard_ham = alpha_hard_ham

        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.index = None
        self.train_metadata = None
        self.class_weights = None
        self.best_alpha = 0.5
        self.model_info = {}

    def _load_model(self):
        """Load the transformer model and tokenizer"""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()

    def load_dataset(self, source='kaggle', file_id=None):
        """Load dataset from Kaggle or Google Drive"""
        if source == 'kaggle':
            return self._load_data_from_kaggle()
        elif source == 'gdrive':
            if file_id is None:
                file_id = "1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R"
            return self._load_data_from_gdrive(file_id)
        else:
            raise ValueError("Source must be 'kaggle' or 'gdrive'")

    def _load_data_from_kaggle(self):
        """Load Vietnamese spam dataset from Kaggle"""
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "victorhoward2/vietnamese-spam-post-in-social-network",
            "vi_dataset.csv"
        )
        return self._preprocess_dataframe(df)

	#! MOVE DATA to /data folder
    def _load_data_from_gdrive(self, file_id):
        """Load dataset from Google Drive"""
        output_path = f"data/gdrive_dataset.csv"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        df = pd.read_csv(output_path)
        return self._preprocess_dataframe(df)

    def _preprocess_dataframe(self, df):
        """Preprocess the loaded dataframe to extract messages and labels"""
        # Try to identify text and label columns
        text_column = None
        label_column = None

        # Common text column names
        text_candidates = ['message', 'text', 'content', 'email', 'post', 'comment', "texts_vi"]
        for col in df.columns:
            if col.lower() in text_candidates or 'text' in col.lower() or 'message' in col.lower():
                text_column = col
                break

        # Common label column names
        label_candidates = ['label', 'class', 'category', 'type']
        for col in df.columns:
            if col.lower() in label_candidates or 'label' in col.lower():
                label_column = col
                break

        # If not found, use first two columns
        if text_column is None:
            text_column = df.columns[0]
        if label_column is None:
            label_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        # Clean text data
        df[text_column] = df[text_column].astype(str).fillna('')
        df = df[df[text_column].str.strip() != '']

        # Clean labels - convert to ham/spam format
        df[label_column] = df[label_column].astype(str).str.lower()

        # Map various label formats to ham/spam
        label_mapping = {
            '0': 'ham', '1': 'spam',
            'ham': 'ham', 'spam': 'spam',
            'normal': 'ham', 'spam': 'spam',
            'legitimate': 'ham', 'phishing': 'spam',
            'not_spam': 'ham', 'is_spam': 'spam'
        }

        df[label_column] = df[label_column].map(label_mapping).fillna(df[label_column])

        messages = df[text_column].tolist()
        labels = df[label_column].tolist()

        return messages, labels

    def _average_pool(self, last_hidden_states, attention_mask):
        """Average pooling for embeddings"""
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embeddings(self, texts, batch_size=32):
        """Generate embeddings for texts"""
        self._load_model()
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_texts_with_prefix = [f"passage: {text}" for text in batch_texts]

            batch_dict = self.tokenizer(
                batch_texts_with_prefix,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                batch_embeddings = self._average_pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"]
                )
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def _calculate_class_weights(self, labels):
        """Calculate class weights for handling imbalanced data"""
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(label_counts)

        class_weights = {}
        for label, count in label_counts.items():
            class_weights[label] = total_samples / (num_classes * count)

        return class_weights

    def _optimize_alpha_parameter(self, test_embeddings, test_metadata, k=10):
        """Find optimal alpha value for best accuracy"""
        alpha_values = np.arange(0.0, 1.1, 0.1)
        best_alpha = 0.0
        best_accuracy = 0.0

        for alpha in alpha_values:
            correct = 0
            total = len(test_embeddings)

            for i in range(total):
                query_embedding = test_embeddings[i:i+1].astype("float32")
                true_label = test_metadata[i]["label"]
                query_text = test_metadata[i]["message"]

                result = self._classify_with_weighted_knn(
                    query_text, query_embedding, k=k, alpha=alpha, explain=False
                )

                if result["prediction"] == true_label:
                    correct += 1

            accuracy = correct / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_alpha = alpha

        return best_alpha

    def _compute_quick_saliency(self, text):
        """Enhanced saliency computation for subtle spam detection"""
        words = text.lower().split()
        text_lower = text.lower()

        basic_spam_keywords = [
            'free', 'click', 'urgent', 'limited', 'offer', 'discount', 'sale', 'win', 'prize',
            'money', 'cash', 'earn', 'guaranteed', 'act now', 'call now', 'congratulations',
            'miễn phí', 'khuyến mãi', 'giảm giá', 'ưu đãi', 'thắng', 'giải thưởng', 'tiền',
            'kiếm tiền', 'đảm bảo', 'hành động ngay', 'chúc mừng', 'cơ hội', 'quà tặng'
        ]

        social_engineering_keywords = [
            'mom', 'boss', 'hr', 'manager', 'security update', 'unusual login',
            'hospital bill', 'emergency', 'help buy', 'reimburse', 'gift cards',
            'mẹ', 'sếp', 'nhân sự', 'cập nhật bảo mật', 'đăng nhập bất thường',
            'viện phí', 'khẩn cấp', 'giúp mua', 'hoàn tiền'
        ]

        urgency_patterns = [
            'today', 'tomorrow', 'this week', 'before friday', 'reply yes',
            'hôm nay', 'ngày mai', 'tuần này', 'trước thứ sáu', 'trả lời có'
        ]

        money_patterns = [
            r'\$\d+', r'\d+\$', r'\d+\s*dollar', r'\d+\s*usd',
            r'\d+\s*triệu', r'\d+\s*nghìn', r'\d+\s*đồng'
        ]

        # Calculate scores
        basic_spam_score = sum(1 for word in words if any(keyword in word for keyword in basic_spam_keywords))
        social_eng_score = sum(2 for keyword in social_engineering_keywords if keyword in text_lower)
        urgency_score = sum(1.5 for pattern in urgency_patterns if pattern in text_lower)

        # Money pattern detection
        money_score = 0
        for pattern in money_patterns:
            if re.search(pattern, text_lower):
                money_score += 2

        # Combined saliency score
        total_score = (basic_spam_score + social_eng_score + urgency_score + money_score)
        saliency = min(1.0, max(0.1, total_score / max(len(words), 1) + 0.2))

        return saliency

    def _compute_saliency_scores(self, query_text, k=10):
        """Compute saliency scores for explainability"""
        self._load_model()
        tokens = self.tokenizer.tokenize(query_text)

        if len(tokens) <= 1:
            return np.array([1.0])

        # Get original embedding and spam score
        query_with_prefix = f"query: {query_text}"
        batch_dict = self.tokenizer([query_with_prefix], max_length=512, padding=True, truncation=True, return_tensors="pt")
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            original_embedding = self._average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            original_embedding = F.normalize(original_embedding, p=2, dim=1)
            original_embedding = original_embedding.cpu().numpy().astype("float32")

        original_scores, original_indices = self.index.search(original_embedding, k)
        original_spam_score = sum(s for s, idx in zip(original_scores[0], original_indices[0])
                                 if self.train_metadata[idx]["label"] == "spam")

        saliencies = []

        # Compute saliency for each token
        for i, token in enumerate(tokens):
            token_mask = tokens.copy()
            token_mask[i] = self.tokenizer.pad_token
            masked_text = self.tokenizer.convert_tokens_to_string(token_mask)

            masked_query = f"query: {masked_text}"
            masked_batch_dict = self.tokenizer([masked_query], max_length=512, padding=True, truncation=True, return_tensors="pt")
            masked_batch_dict = {k: v.to(self.device) for k, v in masked_batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**masked_batch_dict)
                masked_embedding = self._average_pool(outputs.last_hidden_state, masked_batch_dict["attention_mask"])
                masked_embedding = F.normalize(masked_embedding, p=2, dim=1)
                masked_embedding = masked_embedding.cpu().numpy().astype("float32")

            masked_scores, masked_indices = self.index.search(masked_embedding, k)
            masked_spam_score = sum(s for s, idx in zip(masked_scores[0], masked_indices[0])
                                   if self.train_metadata[idx]["label"] == "spam")

            saliency = original_spam_score - masked_spam_score
            saliencies.append(saliency)

        # Normalize saliencies
        arr = np.array(saliencies)
        if len(arr) > 1:
            arr = (arr - arr.min()) / (np.ptp(arr) + 1e-12)
        else:
            arr = np.array([1.0])

        return arr

    def _classify_with_weighted_knn(self, query_text, query_embedding=None, k=10, alpha=0.5, explain=False):
        """Enhanced KNN classification with custom weighting formula"""
        self._load_model()

        # Get query embedding if not provided
        if query_embedding is None:
            query_with_prefix = f"query: {query_text}"
            batch_dict = self.tokenizer([query_with_prefix], max_length=512, padding=True, truncation=True, return_tensors="pt")
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                query_embedding = self._average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                query_embedding = F.normalize(query_embedding, p=2, dim=1)
                query_embedding = query_embedding.cpu().numpy().astype("float32")

        # Get nearest neighbors
        scores, indices = self.index.search(query_embedding, k)

        # Compute saliency weight
        if explain:
            saliency_scores = self._compute_saliency_scores(query_text, k)
            saliency_weight = float(np.mean(saliency_scores))  # Convert to Python float
            tokens = self.tokenizer.tokenize(query_text)
        else:
            saliency_weight = self._compute_quick_saliency(query_text)
            saliency_scores = None
            tokens = None

        # Calculate weighted votes
        vote_scores = {"ham": 0.0, "spam": 0.0}
        neighbor_info = []

        for i in range(k):
            neighbor_idx = int(indices[0][i])  # Convert to Python int
            similarity = float(scores[0][i])  # Convert to Python float
            neighbor_label = self.train_metadata[neighbor_idx]["label"]
            neighbor_message = self.train_metadata[neighbor_idx]["message"]

            # Apply custom weighting formula
            weight = (1 - alpha) * similarity * self.class_weights[neighbor_label] + alpha * saliency_weight
            if weight == 0:
                vote_scores[neighbor_label] += 1
            else:
                vote_scores[neighbor_label] += weight

            neighbor_info.append({
                "score": float(similarity),  # Ensure Python float
                "weight": float(weight),     # Ensure Python float
                "label": neighbor_label,
                "message": neighbor_message[:100] + "..." if len(neighbor_message) > 100 else neighbor_message
            })

        # Get prediction
        predicted_label = max(vote_scores.keys(), key=lambda k: vote_scores[k])

        result = {
            "prediction": predicted_label,
            "vote_scores": {k: float(v) for k, v in vote_scores.items()},  # Convert to Python float
            "neighbors": neighbor_info,
            "saliency_weight": float(saliency_weight),  # Ensure Python float
            "alpha": float(alpha)  # Ensure Python float
        }

        if explain:
            result["tokens"] = tokens
            result["saliency_scores"] = [float(x) for x in saliency_scores] if saliency_scores is not None else None

        return result

    def _classify_spam_subcategory(self, spam_texts):
        """Classify spam into subcategories"""
        if not spam_texts:
            return []

        subcategories = []

        # Define category keywords
        category_keywords = {
            'spam_quangcao': [
                'khuyến mãi', 'giảm giá', 'sale', 'ưu đãi', 'mua ngay', 'giá rẻ', 'miễn phí',
                'quà tặng', 'voucher', 'coupon', 'giải thưởng', 'trúng thưởng', 'cơ hội', 'trúng',
                'discount', 'sale', 'offer', 'promotion', 'free', 'deal', 'buy now', 'limited time',
                'special offer', 'bargain', 'cheap', 'save money', 'win', 'prize', 'gift', 'won',
                'congratulations', 'claim', 'click here', '$', 'money', 'cash'
            ],
            'spam_hethong': [
                'thông báo', 'cảnh báo', 'tài khoản', 'bảo mật', 'xác nhận', 'cập nhật',
                'hệ thống', 'đăng nhập', 'mật khẩu', 'bị khóa', 'hết hạn', 'gia hạn', 'khóa',
                'notification', 'alert', 'account', 'security', 'confirm', 'update',
                'system', 'login', 'password', 'locked', 'expired', 'renewal', 'verify',
                'suspended', 'warning', 'breach', 'urgent', 'immediately'
            ]
        }

        for text in spam_texts:
            text_lower = text.lower()

            # Score each category
            category_scores = {}
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                category_scores[category] = score

            # Classify based on highest score
            if max(category_scores.values()) == 0:
                subcategories.append('spam_khac')
            else:
                best_category = max(category_scores, key=category_scores.get)
                subcategories.append(best_category)

        return subcategories

    def train(self, messages, labels, test_size=0.2, progress_callback=None):
        """Train the spam classification model"""
        if progress_callback:
            progress_callback(0.1, "Initializing model...")

        self._load_model()

        # Prepare data
        le = LabelEncoder()
        y = le.fit_transform(labels)

        if progress_callback:
            progress_callback(0.2, "Generating embeddings...")

        # Generate embeddings
        X_embeddings = self.get_embeddings(messages)

        # Create metadata - FIX: Convert numpy types to Python native types
        metadata = [{"index": int(i), "message": message, "label": label, "label_encoded": int(y[i])}
                    for i, (message, label) in enumerate(zip(messages, labels))]

        if progress_callback:
            progress_callback(0.5, "Splitting data...")

        # Train-test split
        X_train_emb, X_test_emb, train_metadata, test_metadata = train_test_split(
            X_embeddings, metadata, test_size=test_size, random_state=42,
            stratify=[m["label"] for m in metadata]
        )

        if progress_callback:
            progress_callback(0.6, "Creating FAISS index...")

        # Create FAISS index
        dimension = X_train_emb.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(X_train_emb.astype("float32"))

        # Store training metadata
        self.train_metadata = train_metadata

        if progress_callback:
            progress_callback(0.7, "Calculating class weights...")

        # Calculate class weights
        train_labels = [m["label"] for m in train_metadata]
        self.class_weights = self._calculate_class_weights(train_labels)

        if progress_callback:
            progress_callback(0.8, "Optimizing parameters...")

        # Optimize alpha parameter
        self.best_alpha = self._optimize_alpha_parameter(X_test_emb, test_metadata)

        if progress_callback:
            progress_callback(0.9, "Evaluating model...")

        # Final evaluation
        accuracy_results = self._evaluate_accuracy(X_test_emb, test_metadata)

        # Store model info - FIX: Convert numpy types
        self.model_info = convert_numpy_types({
            'dataset_size': len(messages),
            'model_name': self.model_name,
            'best_alpha': self.best_alpha,
            'training_date': datetime.now().isoformat(),
            'accuracy_results': accuracy_results
        })

        if progress_callback:
            progress_callback(1.0, "Training completed!")

        return {
            'best_alpha': float(self.best_alpha),  # Convert to Python float
            'accuracy_results': convert_numpy_types(accuracy_results),
            'class_weights': convert_numpy_types(self.class_weights)
        }

    def _evaluate_accuracy(self, test_embeddings, test_metadata, k_values=[1, 3, 5]):
        """Evaluate accuracy using weighted KNN classification"""
        results = {}

        for k in k_values:
            correct = 0
            total = len(test_embeddings)

            for i in range(total):
                query_text = test_metadata[i]["message"]
                true_label = test_metadata[i]["label"]
                query_embedding = test_embeddings[i:i+1].astype("float32")

                result = self._classify_with_weighted_knn(
                    query_text, query_embedding, k=k, alpha=self.best_alpha, explain=False
                )

                if result["prediction"] == true_label:
                    correct += 1

            accuracy = correct / total
            results[k] = float(accuracy)  # Convert to Python float

        return results

    def run_enhanced_pipeline(self, messages, labels, test_size=0.2, use_augmentation=True):
        """Run the complete enhanced spam classification pipeline with data augmentation"""

        print("=== Enhanced Spam Classification Pipeline ===")

        # 1. Prepare data
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # ===== DATA AUGMENTATION =====
        if use_augmentation:
            print("\n=== Data Augmentation ===")
            try:
                augmented_messages, augmented_labels = enhanced_augmentation(
                    messages, labels,
                    aug_ratio=self.aug_ratio,
                    alpha_hard_ham=self.alpha_hard_ham,
                )

                if augmented_messages:  # Only add if data was generated
                    # Combine original + augmented
                    original_count = len(messages)
                    messages = messages + augmented_messages  # Extend list
                    labels = labels + augmented_labels       # Extend list

                    print(f"📈 Dataset size: {original_count} → {len(messages)} (+{len(augmented_messages)})")

                    # Re-encode labels with augmented data
                    y = le.fit_transform(labels)
                else:
                    print("ℹ️ No augmented data generated")
            except Exception as e:
                print(f"⚠️ Augmentation failed: {e}")
                print("ℹ️ Continuing with original data...")
        else:
            print("ℹ️ Data augmentation disabled")

        # 2. Load model and generate embeddings
        print("Generating embeddings...")
        self._load_model()
        X_embeddings = self.get_embeddings(messages)

        # 3. Create metadata
        metadata = []
        for i, (message, label) in enumerate(zip(messages, labels)):
            # Simple label encoding: ham=0, spam=1
            label_encoded = 1 if label == 'spam' else 0

            metadata.append({
                "index": i,
                "message": message,
                "label": label,
                "label_encoded": label_encoded
            })

        # 4. Train-test split
        X_train_emb, X_test_emb, train_metadata, test_metadata = train_test_split(
            X_embeddings, metadata, test_size=test_size, random_state=42,
            stratify=[m["label"] for m in metadata]
        )

        # 5. Create FAISS index
        print("Creating FAISS index...")
        dimension = X_train_emb.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        # Ensure proper numpy array format for FAISS
        if not isinstance(X_train_emb, np.ndarray):
            X_train_emb = np.array(X_train_emb)
        X_train_emb_float32 = X_train_emb.astype(np.float32)
        self.index.add(X_train_emb_float32)

        # Store training metadata
        self.train_metadata = train_metadata

        # 6. Calculate class weights
        train_labels = [m["label"] for m in train_metadata]
        self.class_weights = self._calculate_class_weights(train_labels)

        # 7. Optimize alpha parameter
        test_labels = [m["label"] for m in test_metadata]
        self.best_alpha = self._optimize_alpha_parameter(X_test_emb, test_metadata)

        # 8. Final evaluation
        print("\n=== Final Evaluation ===")
        accuracy_results = self._evaluate_accuracy(X_test_emb, test_metadata)

        # Print results
        print("Accuracy by k-value:")
        for k, acc in accuracy_results.items():
            print(f"  k={k}: {acc:.4f} ({acc*100:.2f}%)")

        # 9. Analyze spam subcategories
        subcat_counts = {}  # Initialize empty dict
        spam_texts = [m["message"] for m in test_metadata if m["label"] == "spam"]
        if spam_texts:
            print(f"\n=== Spam Subcategory Analysis ===")
            spam_subcategories = self._classify_spam_subcategory(spam_texts)
            subcat_counts = Counter(spam_subcategories)

            print("Spam subcategory distribution:")
            for subcat, count in subcat_counts.items():
                print(f"  {subcat}: {count} ({count/len(spam_texts)*100:.1f}%)")

        # 10. Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "dataset_size": len(X_embeddings),
            "train_size": len(X_train_emb),
            "test_size": len(X_test_emb),
            "best_alpha": float(self.best_alpha),
            "accuracy_results": convert_numpy_types(accuracy_results),
            "class_weights": convert_numpy_types(self.class_weights),
            "spam_subcategories": dict(subcat_counts) if spam_texts else {}
        }

        # Store model info
        self.model_info = results

        with open("enhanced_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n*** Results saved to enhanced_results.json ***")

        return {
            "index": self.index,
            "train_metadata": self.train_metadata,
            "test_metadata": test_metadata,
            "class_weights": self.class_weights,
            "best_alpha": self.best_alpha,
            "results": results,
        }

    def classify_message(self, message, k=5, explain=False):
        """Classify a single message"""
        # FIX: Better check for model and index existence
        if self.model is None or self.index is None:
            raise ValueError("Model not trained. Please train the model first.")

        # FIX: Validate input
        if not message or not isinstance(message, str):
            raise ValueError("Message must be a non-empty string")

        message = str(message).strip()
        if not message:
            raise ValueError("Message cannot be empty after stripping")

        try:
            # Get prediction
            result = self._classify_with_weighted_knn(
                message, k=k, alpha=self.best_alpha, explain=explain
            )

            # If spam, classify subcategory
            if result["prediction"] == "spam":
                subcategories = self._classify_spam_subcategory([message])
                result["subcategory"] = subcategories[0] if subcategories else "spam_khac"

            return result

        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")

    def save_to_files(self):
        # Save FAISS index
        faiss.write_index(self.index, f"{self.save_folder_path}/faiss_index.bin")

        # Save metadata and weights - FIX: Convert numpy types before saving
        with open(f"{self.save_folder_path}/train_metadata.json", "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(self.train_metadata), f, ensure_ascii=False, indent=2)

        with open(f"{self.save_folder_path}/class_weights.json", "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(self.class_weights), f, indent=2)

        with open(f"{self.save_folder_path}/model_config.json", "w", encoding="utf-8") as f:
            config = convert_numpy_types({
                'model_name': self.model_name,
                'best_alpha': self.best_alpha,
                'model_info': self.model_info
            })
            json.dump(config, f, indent=2)

        # Save other artifacts
        artifacts = {
            'device': str(self.device)
        }

        with open(f"{self.save_folder_path}/model_artifacts.pkl", "wb") as f:
            pickle.dump(artifacts, f)

    #? Change @classmethod to Object Method
    def load_from_files(self): #? set folder_path in app.py
        """Load model from saved files"""
        print('path:', self.save_folder_path)

        # Load config
        with open(f"{self.save_folder_path}/model_config.json", "r") as f:
            """
            config = {
                'model_name': 'intfloat/multilingual-e5-base',
                'best_alpha': 1.0,
                'model_info': {
                    'dataset_size': 5572,
                    'model_name': 'intfloat/multilingual-e5-base',
                    'best_alpha': 1.0,
                    'training_date': '2025-08-08T20:32:16.059342',
                    'accuracy_results': {'1': 0.9901345291479821, '3': 0.9919282511210762, '5': 0.9910313901345291}
                }
            }
            """
            config = json.load(f)

        # load config to SpamClassifier Object's attribute
        self.model_name = config['model_name']
        self.best_alpha = config['best_alpha']
        self.model_info = config['model_info']

        # Load model components
        self._load_model()

        # Load FAISS index
        self.index = faiss.read_index(f"{self.save_folder_path}/faiss_index.bin")

        # Load metadata and weights
        with open(f"{self.save_folder_path}/train_metadata.json", "r", encoding="utf-8") as f:
            self.train_metadata = json.load(f)

        with open(f"{self.save_folder_path}/class_weights.json", "r") as f:
            self.class_weights = json.load(f)

        print(f"Model loaded successfully from {self.save_folder_path}")
        return self