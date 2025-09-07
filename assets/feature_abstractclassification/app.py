# Thêm vào đầu file - Memory và Error Handling
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gc  # Garbage collector để giải phóng memory
import psutil  # Monitor memory usage
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Literal, Union
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from scipy.sparse import issparse

# Thêm cấu hình memory
st.set_page_config(
    page_title="Scientific Text Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hàm monitor memory
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

# Caching functions được sửa lại với error handling
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset with error handling"""
    try:
        with st.spinner("📚 Loading arxiv dataset..."):
            # Giảm sample size để tránh memory issues
            SAMPLE_SIZE = 500  # Giảm từ 1000 xuống 500
            
            # Monitor memory
            initial_memory = get_memory_usage()
            st.sidebar.info(f"Initial Memory: {initial_memory:.1f} MB")
            
            ds = load_dataset("UniverseTBD/arxiv-abstracts-large")
            
            # Sample data với batch processing
            samples = []
            processed_count = 0
            
            for s in ds['train']:
                if len(s['categories'].split(' ')) != 1:
                    continue

                cur_category = s['categories'].strip().split('.')[0]
                if cur_category not in CATEGORIES_TO_SELECT:
                    continue

                samples.append(s)
                processed_count += 1
                
                # Update progress
                if processed_count % 100 == 0:
                    current_memory = get_memory_usage()
                    st.sidebar.info(f"Processed: {processed_count}, Memory: {current_memory:.1f} MB")
                
                if len(samples) >= SAMPLE_SIZE:
                    break

            # Preprocess với memory optimization
            preprocessed_samples = []
            for i, s in enumerate(samples):
                try:
                    abstract = s['abstract']
                    abstract = abstract.strip().replace("\n", " ")
                    abstract = re.sub(r'[^\w\s]', "", abstract)
                    abstract = re.sub(r'\d+', "", abstract)
                    abstract = re.sub(r'\s+', " ", abstract)
                    abstract = abstract.lower()

                    # Giới hạn độ dài text để tránh memory issues
                    if len(abstract) > 1000:
                        abstract = abstract[:1000]

                    part = s["categories"].split(" ")
                    category = part[0].split(".")[0]

                    preprocessed_samples.append({
                        "text": abstract,
                        "label": category
                    })
                    
                    # Update progress
                    if (i + 1) % 100 == 0:
                        progress = (i + 1) / len(samples)
                        st.sidebar.progress(progress)
                        
                except Exception as e:
                    st.warning(f"Error processing sample {i}: {str(e)}")
                    continue

            # Clean up
            del ds, samples
            gc.collect()
            
            final_memory = get_memory_usage()
            st.sidebar.success(f"Data loaded! Final Memory: {final_memory:.1f} MB")
            
            return preprocessed_samples
            
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return []

@st.cache_data
def create_vectorizers(preprocessed_samples):
    """Create different types of vectorizers with memory optimization"""
    try:
        with st.spinner("🔧 Creating vectorizers..."):
            if not preprocessed_samples:
                st.error("No data to process!")
                return None
                
            # Monitor memory
            initial_memory = get_memory_usage()
            st.sidebar.info(f"Vectorizer Memory Start: {initial_memory:.1f} MB")
            
            # Create label mappings
            sorted_categories = sorted(CATEGORIES_TO_SELECT, key=lambda x: x.lower())
            label_to_id = {label: i for i, label in enumerate(sorted_categories)}
            id_to_label = {i: label for i, label in enumerate(sorted_categories)}

            # Prepare data
            X_full = [s["text"] for s in preprocessed_samples]
            y_full = [label_to_id[s["label"]] for s in preprocessed_samples]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )

            # Tạo BOW và TF-IDF trước
            st.sidebar.info("Creating BOW vectorizer...")
            bow_vectorizer = CountVectorizer(max_features=5000)  # Giới hạn features
            X_train_bow = bow_vectorizer.fit_transform(X_train).toarray()
            X_test_bow = bow_vectorizer.transform(X_test).toarray()
            
            bow_memory = get_memory_usage()
            st.sidebar.info(f"BOW Memory: {bow_memory:.1f} MB")

            st.sidebar.info("Creating TF-IDF vectorizer...")
            tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Giới hạn features
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
            X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()
            
            tfidf_memory = get_memory_usage()
            st.sidebar.info(f"TF-IDF Memory: {tfidf_memory:.1f} MB")

            # Embeddings với batch processing
            st.sidebar.info("Creating embeddings (this may take a while)...")
            
            try:
                embedding_vectorizer = EmbeddingVectorizer()
                
                # Process embeddings in batches
                batch_size = 32
                X_train_embeddings = []
                
                for i in range(0, len(X_train), batch_size):
                    batch = X_train[i:i+batch_size]
                    batch_embeddings = embedding_vectorizer.transform(batch)
                    X_train_embeddings.extend(batch_embeddings)
                    
                    # Update progress
                    progress = min((i + batch_size) / len(X_train), 1.0)
                    st.sidebar.progress(progress)
                    
                    # Memory check
                    current_memory = get_memory_usage()
                    if current_memory > 4000:  # Nếu memory > 4GB
                        st.warning("Memory usage too high! Switching to smaller batch size.")
                        batch_size = 16
                
                X_train_embeddings = np.array(X_train_embeddings)
                
                # Test embeddings
                X_test_embeddings = []
                for i in range(0, len(X_test), batch_size):
                    batch = X_test[i:i+batch_size]
                    batch_embeddings = embedding_vectorizer.transform(batch)
                    X_test_embeddings.extend(batch_embeddings)
                
                X_test_embeddings = np.array(X_test_embeddings)
                
                embed_memory = get_memory_usage()
                st.sidebar.success(f"Embeddings Memory: {embed_memory:.1f} MB")
                
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                # Fallback: skip embeddings
                X_train_embeddings = np.zeros((len(X_train), 384))
                X_test_embeddings = np.zeros((len(X_test), 384))
                embedding_vectorizer = None

            # Clean up
            gc.collect()
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_train_bow': X_train_bow,
                'X_test_bow': X_test_bow,
                'X_train_tfidf': X_train_tfidf,
                'X_test_tfidf': X_test_tfidf,
                'X_train_embeddings': X_train_embeddings,
                'X_test_embeddings': X_test_embeddings,
                'bow_vectorizer': bow_vectorizer,
                'tfidf_vectorizer': tfidf_vectorizer,
                'embedding_vectorizer': embedding_vectorizer,
                'label_to_id': label_to_id,
                'id_to_label': id_to_label
            }
            
    except Exception as e:
        st.error(f"❌ Error creating vectorizers: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return None

@st.cache_data
def train_all_models(data_dict):
    """Train all model combinations with memory optimization"""
    try:
        if data_dict is None:
            st.error("No data available for training!")
            return []
            
        with st.spinner("🤖 Training all models..."):
            initial_memory = get_memory_usage()
            st.sidebar.info(f"Training Memory Start: {initial_memory:.1f} MB")
            
            # Giảm số models để tránh memory issues
            models = {
                'Naive Bayes': GaussianNB(),
                'KNN (Majority)': CustomKNN(n_neighbors=5, voting='majority'),
                'KNN (Weighted)': CustomKNN(n_neighbors=5, voting='weighted'),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
            }

            datasets = {
                'Bag of Words': (data_dict['X_train_bow'], data_dict['X_test_bow']),
                'TF-IDF': (data_dict['X_train_tfidf'], data_dict['X_test_tfidf']),
            }
            
            # Chỉ thêm embeddings nếu có
            if data_dict['embedding_vectorizer'] is not None:
                datasets['Embeddings'] = (data_dict['X_train_embeddings'], data_dict['X_test_embeddings'])

            results = []
            total_combinations = len(models) * len(datasets)
            current_combination = 0
            
            for vectorizer_name, (X_train_vec, X_test_vec) in datasets.items():
                for model_name, model in models.items():
                    try:
                        current_combination += 1
                        progress = current_combination / total_combinations
                        st.sidebar.progress(progress)
                        st.sidebar.info(f"Training: {model_name} - {vectorizer_name}")
                        
                        # Create fresh model copy
                        if 'Naive Bayes' in model_name:
                            model_copy = GaussianNB()
                        elif 'KNN' in model_name:
                            if 'Majority' in model_name:
                                model_copy = CustomKNN(n_neighbors=5, voting='majority')
                            elif 'Weighted' in model_name:
                                model_copy = CustomKNN(n_neighbors=5, voting='weighted')
                        else:
                            model_copy = DecisionTreeClassifier(random_state=42, max_depth=10)

                        # Train and evaluate
                        model_copy.fit(X_train_vec, data_dict['y_train'])
                        y_pred = model_copy.predict(X_test_vec)
                        accuracy = accuracy_score(data_dict['y_test'], y_pred)

                        # Calculate classification report
                        unique_labels = sorted(list(set(data_dict['y_train']) | set(data_dict['y_test']) | set(y_pred)))
                        target_names = [data_dict['id_to_label'][label] for label in unique_labels]

                        report = classification_report(
                            data_dict['y_test'], y_pred,
                            labels=unique_labels,
                            target_names=target_names,
                            output_dict=True,
                            zero_division=0
                        )

                        cm = confusion_matrix(data_dict['y_test'], y_pred, labels=unique_labels)

                        results.append({
                            'Model': model_name,
                            'Vectorizer': vectorizer_name,
                            'Accuracy': accuracy,
                            'Report': report,
                            'Confusion_Matrix': cm,
                            'Predictions': y_pred,
                            'Trained_Model': model_copy
                        })
                        
                        # Memory check
                        current_memory = get_memory_usage()
                        st.sidebar.info(f"Current Memory: {current_memory:.1f} MB")
                        
                        # Clean up
                        del model_copy
                        gc.collect()
                        
                    except Exception as e:
                        st.warning(f"Error training {model_name} - {vectorizer_name}: {str(e)}")
                        continue

            final_memory = get_memory_usage()
            st.sidebar.success(f"Training Complete! Final Memory: {final_memory:.1f} MB")
            
            return results
            
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return []

# Main app logic với error handling
def main():
    try:
        # Sidebar controls
        st.sidebar.markdown("### 📊 Analysis Options")
        
        # Memory monitor
        current_memory = get_memory_usage()
        st.sidebar.metric("Current Memory Usage", f"{current_memory:.1f} MB")
        
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ["📈 Model Performance Overview", "🎯 Model Comparison", "🔍 Detailed Analysis", "📝 Text Classification Demo"]
        )

        # Load data với step-by-step approach
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            
        if not st.session_state.data_loaded:
            if st.button("🚀 Start Data Loading", type="primary"):
                try:
                    # Step 1: Load data
                    st.info("Step 1/3: Loading and preprocessing data...")
                    preprocessed_samples = load_and_preprocess_data()
                    
                    if not preprocessed_samples:
                        st.error("Failed to load data. Please try again.")
                        return
                    
                    # Step 2: Create vectorizers
                    st.info("Step 2/3: Creating vectorizers...")
                    data_dict = create_vectorizers(preprocessed_samples)
                    
                    if data_dict is None:
                        st.error("Failed to create vectorizers. Please try again.")
                        return
                    
                    # Step 3: Train models
                    st.info("Step 3/3: Training models...")
                    results = train_all_models(data_dict)
                    
                    if not results:
                        st.error("Failed to train models. Please try again.")
                        return
                    
                    # Save to session state
                    st.session_state.preprocessed_samples = preprocessed_samples
                    st.session_state.data_dict = data_dict
                    st.session_state.results = results
                    st.session_state.data_loaded = True
                    
                    st.success("✅ Data loaded and models trained successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error during initialization: {str(e)}")
                    st.error("Traceback:")
                    st.code(traceback.format_exc())
                    return
        else:
            # Analysis sections
            if analysis_type == "📈 Model Performance Overview":
                show_model_overview()
            elif analysis_type == "🎯 Model Comparison":
                show_model_comparison()
            elif analysis_type == "🔍 Detailed Analysis":
                show_detailed_analysis()
            elif analysis_type == "📝 Text Classification Demo":
                show_text_classification_demo()
                
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()