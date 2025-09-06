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

# Page configuration
st.set_page_config(
    page_title="Scientific Text Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🔬 Scientific Text Classification Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## ⚙️ Configuration")

# Constants
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
CACHE_DIR = "./cache"

# Custom KNN Class
class CustomKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, voting='majority', alpha=0.5, metric='cosine'):
        self.n_neighbors = n_neighbors
        self.voting = voting
        self.alpha = alpha
        self.metric = metric

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_train_ = X
        self.y_train_ = y

        # Calculate class weights (inverse frequency)
        class_counts = Counter(y)
        total_samples = len(y)
        self.class_weights_ = {
            label: total_samples / (len(self.classes_) * count)
            for label, count in class_counts.items()
        }
        return self

    def _calculate_saliency_weights(self, X_test):
        feature_vars = np.var(self.X_train_, axis=0)
        feature_vars = np.where(feature_vars == 0, 1e-10, feature_vars)
        saliency = 1.0 / (1.0 + feature_vars)

        saliency_weights = []
        for x_test in X_test:
            weighted_distance = np.sum(saliency * (x_test - np.mean(self.X_train_, axis=0))**2)
            saliency_weights.append(1.0 / (1.0 + weighted_distance))

        return np.array(saliency_weights)

    def predict(self, X):
        X = check_array(X)

        if self.voting == 'majority':
            return self._predict_majority(X)
        elif self.voting == 'weighted':
            return self._predict_weighted(X)
        elif self.voting == 'custom':
            return self._predict_custom(X)
        else:
            raise ValueError(f"Unknown voting scheme: {self.voting}")

    def _predict_majority(self, X):
        predictions = []
        for x in X:
            if self.metric == 'cosine':
                similarities = 1 - cdist([x], self.X_train_, metric='cosine')[0]
                distances = 1 - similarities
            else:
                distances = cdist([x], self.X_train_, metric=self.metric)[0]

            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_labels = self.y_train_[neighbor_indices]
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

    def _predict_weighted(self, X):
        predictions = []
        for x in X:
            if self.metric == 'cosine':
                similarities = 1 - cdist([x], self.X_train_, metric='cosine')[0]
                distances = 1 - similarities
            else:
                distances = cdist([x], self.X_train_, metric=self.metric)[0]

            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_distances = distances[neighbor_indices]
            neighbor_labels = self.y_train_[neighbor_indices]

            weights = 1.0 / (neighbor_distances + 1e-10)
            label_weights = {}
            for label, weight in zip(neighbor_labels, weights):
                label_weights[label] = label_weights.get(label, 0) + weight

            best_label = max(label_weights, key=label_weights.get)
            predictions.append(best_label)

        return np.array(predictions)

    def _predict_custom(self, X):
        predictions = []
        saliency_weights = self._calculate_saliency_weights(X)

        for i, x in enumerate(X):
            if self.metric == 'cosine':
                similarities = 1 - cdist([x], self.X_train_, metric='cosine')[0]
            else:
                distances = cdist([x], self.X_train_, metric=self.metric)[0]
                similarities = 1.0 / (1.0 + distances)

            neighbor_indices = np.argsort(similarities)[::-1][:self.n_neighbors]
            neighbor_similarities = similarities[neighbor_indices]
            neighbor_labels = self.y_train_[neighbor_indices]

            label_weights = {}
            for similarity, label in zip(neighbor_similarities, neighbor_labels):
                custom_weight = (
                    (1 - self.alpha) * similarity * self.class_weights_[label] +
                    self.alpha * saliency_weights[i]
                )
                label_weights[label] = label_weights.get(label, 0) + custom_weight

            best_label = max(label_weights, key=label_weights.get)
            predictions.append(best_label)

        return np.array(predictions)

# Embedding Vectorizer Class
class EmbeddingVectorizer:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def _format_inputs(self, texts: List[str], mode: Literal["query", "passage"]):
        if mode not in {"query", "passage"}:
            raise ValueError("Mode must be either 'query' or 'passage'")
        return [f"{mode}: {t.strip()}" for t in texts]

    def transform(self, texts: List[str], mode: Literal["query", "passage"] = "query"):
        if mode == "raw":
            inputs = texts
        else:
            inputs = self._format_inputs(texts, mode)

        embeddings = self.model.encode(inputs, normalize_embeddings=self.normalize)
        return embeddings.tolist()

# Caching functions
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    with st.spinner("📚 Loading arxiv dataset..."):
        ds = load_dataset("UniverseTBD/arxiv-abstracts-large")
        
        # Sample data
        samples = []
        for s in ds['train']:
            if len(s['categories'].split(' ')) != 1:
                continue

            cur_category = s['categories'].strip().split('.')[0]
            if cur_category not in CATEGORIES_TO_SELECT:
                continue

            samples.append(s)
            if len(samples) >= 1000:
                break

        # Preprocess
        preprocessed_samples = []
        for s in samples:
            abstract = s['abstract']
            abstract = abstract.strip().replace("\n", " ")
            abstract = re.sub(r'[^\w\s]', "", abstract)
            abstract = re.sub(r'\d+', "", abstract)
            abstract = re.sub(r'\s+', " ", abstract)
            abstract = abstract.lower()

            part = s["categories"].split(" ")
            category = part[0].split(".")[0]

            preprocessed_samples.append({
                "text": abstract,
                "label": category
            })

        return preprocessed_samples

@st.cache_data
def create_vectorizers(preprocessed_samples):
    """Create different types of vectorizers"""
    with st.spinner("🔧 Creating vectorizers..."):
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

        # Create vectorizers
        bow_vectorizer = CountVectorizer()
        X_train_bow = bow_vectorizer.fit_transform(X_train).toarray()
        X_test_bow = bow_vectorizer.transform(X_test).toarray()

        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

        embedding_vectorizer = EmbeddingVectorizer()
        X_train_embeddings = np.array(embedding_vectorizer.transform(X_train))
        X_test_embeddings = np.array(embedding_vectorizer.transform(X_test))

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

@st.cache_data
def train_all_models(data_dict):
    """Train all model combinations"""
    with st.spinner("🤖 Training all models..."):
        models = {
            'Naive Bayes': GaussianNB(),
            'KNN (Majority)': CustomKNN(n_neighbors=5, voting='majority'),
            'KNN (Weighted)': CustomKNN(n_neighbors=5, voting='weighted'),
            'KNN (Custom α=0.3)': CustomKNN(n_neighbors=5, voting='custom', alpha=0.3),
            'KNN (Custom α=0.5)': CustomKNN(n_neighbors=5, voting='custom', alpha=0.5),
            'KNN (Custom α=0.7)': CustomKNN(n_neighbors=5, voting='custom', alpha=0.7),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
        }

        datasets = {
            'Bag of Words': (data_dict['X_train_bow'], data_dict['X_test_bow']),
            'TF-IDF': (data_dict['X_train_tfidf'], data_dict['X_test_tfidf']),
            'Embeddings': (data_dict['X_train_embeddings'], data_dict['X_test_embeddings'])
        }

        results = []
        
        for vectorizer_name, (X_train_vec, X_test_vec) in datasets.items():
            for model_name, model in models.items():
                # Create fresh model copy
                if 'Naive Bayes' in model_name:
                    model_copy = GaussianNB()
                elif 'KNN' in model_name:
                    if 'Majority' in model_name:
                        model_copy = CustomKNN(n_neighbors=5, voting='majority')
                    elif 'Weighted' in model_name:
                        model_copy = CustomKNN(n_neighbors=5, voting='weighted')
                    elif 'α=0.3' in model_name:
                        model_copy = CustomKNN(n_neighbors=5, voting='custom', alpha=0.3)
                    elif 'α=0.5' in model_name:
                        model_copy = CustomKNN(n_neighbors=5, voting='custom', alpha=0.5)
                    elif 'α=0.7' in model_name:
                        model_copy = CustomKNN(n_neighbors=5, voting='custom', alpha=0.7)
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

        return results

# Main app logic
def main():
    # Sidebar controls
    st.sidebar.markdown("### 📊 Analysis Options")
    
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["📈 Model Performance Overview", "🎯 Model Comparison", "🔍 Detailed Analysis", "📝 Text Classification Demo"]
    )

    # Load data
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        
    if not st.session_state.data_loaded:
        with st.spinner("🚀 Initializing application..."):
            preprocessed_samples = load_and_preprocess_data()
            data_dict = create_vectorizers(preprocessed_samples)
            results = train_all_models(data_dict)
            
            st.session_state.preprocessed_samples = preprocessed_samples
            st.session_state.data_dict = data_dict
            st.session_state.results = results
            st.session_state.data_loaded = True
            
        st.success("✅ Data loaded and models trained successfully!")

    # Analysis sections
    if analysis_type == "📈 Model Performance Overview":
        show_model_overview()
    elif analysis_type == "🎯 Model Comparison":
        show_model_comparison()
    elif analysis_type == "🔍 Detailed Analysis":
        show_detailed_analysis()
    elif analysis_type == "📝 Text Classification Demo":
        show_text_classification_demo()

def show_model_overview():
    st.markdown('<h2 class="sub-header">📈 Model Performance Overview</h2>', unsafe_allow_html=True)
    
    # Get results
    results = st.session_state.results
    df_results = pd.DataFrame(results)
    
    # Best model overall
    best_result = max(results, key=lambda x: x['Accuracy'])
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🏆 Best Model</h3>
            <p>{best_result['Model']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🎯 Best Accuracy</h3>
            <p>{best_result['Accuracy']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Best Vectorizer</h3>
            <p>{best_result['Vectorizer']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        total_models = len(results)
        st.markdown(f"""
        <div class="metric-container">
            <h3>🔢 Total Models</h3>
            <p>{total_models}</p>
        </div>
        """, unsafe_allow_html=True)

    # Performance by vectorizer
    st.markdown("### 📊 Performance by Vectorizer")
    
    # Create interactive plot
    fig = px.bar(
        df_results, 
        x='Model', 
        y='Accuracy', 
        color='Vectorizer',
        title="Model Performance Comparison",
        height=500
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.markdown("### 📋 Summary Results")
    summary_df = df_results.pivot_table(
        values='Accuracy', 
        index='Model', 
        columns='Vectorizer'
    ).round(4)
    st.dataframe(summary_df, use_container_width=True)

def show_model_comparison():
    st.markdown('<h2 class="sub-header">🎯 Model Comparison</h2>', unsafe_allow_html=True)
    
    results = st.session_state.results
    
    # Select models to compare
    model_names = list(set([r['Model'] for r in results]))
    vectorizer_names = list(set([r['Vectorizer'] for r in results]))
    
    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect(
            "Select Models to Compare:", 
            model_names, 
            default=model_names[:3]
        )
    with col2:
        selected_vectorizer = st.selectbox(
            "Select Vectorizer:", 
            vectorizer_names
        )
    
    if selected_models and selected_vectorizer:
        # Filter results
        filtered_results = [
            r for r in results 
            if r['Model'] in selected_models and r['Vectorizer'] == selected_vectorizer
        ]
        
        if filtered_results:
            # Accuracy comparison
            st.markdown("### 🎯 Accuracy Comparison")
            
            accuracies = [r['Accuracy'] for r in filtered_results]
            models = [r['Model'] for r in filtered_results]
            
            fig = go.Figure(data=[
                go.Bar(x=models, y=accuracies, 
                       text=[f'{acc:.4f}' for acc in accuracies],
                       textposition='auto')
            ])
            fig.update_layout(
                title=f"Accuracy Comparison - {selected_vectorizer}",
                xaxis_title="Models",
                yaxis_title="Accuracy"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.markdown("### 📊 Detailed Metrics")
            
            comparison_data = []
            for result in filtered_results:
                report = result['Report']
                comparison_data.append({
                    'Model': result['Model'],
                    'Accuracy': result['Accuracy'],
                    'Macro Avg F1': report['macro avg']['f1-score'],
                    'Weighted Avg F1': report['weighted avg']['f1-score'],
                    'Macro Avg Precision': report['macro avg']['precision'],
                    'Macro Avg Recall': report['macro avg']['recall']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.round(4), use_container_width=True)

def show_detailed_analysis():
    st.markdown('<h2 class="sub-header">🔍 Detailed Analysis</h2>', unsafe_allow_html=True)
    
    results = st.session_state.results
    data_dict = st.session_state.data_dict
    
    # Select specific model for detailed analysis
    model_options = [f"{r['Model']} - {r['Vectorizer']}" for r in results]
    selected_model = st.selectbox("Select Model for Detailed Analysis:", model_options)
    
    if selected_model:
        # Find selected result
        model_name, vectorizer_name = selected_model.split(" - ")
        selected_result = next(
            r for r in results 
            if r['Model'] == model_name and r['Vectorizer'] == vectorizer_name
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Classification Report")
            
            report = selected_result['Report']
            class_names = CATEGORIES_TO_SELECT
            
            # Create classification report dataframe
            report_data = []
            for class_name in class_names:
                if class_name in report:
                    report_data.append({
                        'Class': class_name,
                        'Precision': report[class_name]['precision'],
                        'Recall': report[class_name]['recall'],
                        'F1-Score': report[class_name]['f1-score'],
                        'Support': report[class_name]['support']
                    })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df.round(4), use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Confusion Matrix")
            
            cm = selected_result['Confusion_Matrix']
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=CATEGORIES_TO_SELECT,
                yticklabels=CATEGORIES_TO_SELECT,
                ax=ax
            )
            ax.set_title(f'Confusion Matrix: {model_name} - {vectorizer_name}')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)

def show_text_classification_demo():
    st.markdown('<h2 class="sub-header">📝 Text Classification Demo</h2>', unsafe_allow_html=True)
    
    data_dict = st.session_state.data_dict
    results = st.session_state.results
    
    # Get best model
    best_result = max(results, key=lambda x: x['Accuracy'])
    best_model = best_result['Trained_Model']
    
    st.info(f"Using best model: **{best_result['Model']}** with **{best_result['Vectorizer']}** (Accuracy: {best_result['Accuracy']:.4f})")
    
    # Text input
    st.markdown("### ✍️ Enter Scientific Abstract")
    
    # Sample texts for each category
    sample_texts = {
        "Astrophysics": "We study the electronic properties of neutron stars using gravitational wave observations. Our results show unusual magnetic field structures that could explain fast radio bursts observed in distant galaxies.",
        "Condensed Matter": "We investigate the electronic properties of novel two-dimensional materials using density functional theory calculations. The quantum mechanical effects show unusual band structures and magnetic properties.",
        "Computer Science": "We propose a new machine learning algorithm for natural language processing. Our neural network architecture achieves state-of-the-art performance on text classification and sentiment analysis tasks.",
        "Mathematics": "We prove a new theorem in algebraic topology concerning the fundamental group of manifolds. The mathematical framework provides insights into geometric structures and homology theory.",
        "Physics": "We study quantum mechanical systems using computational physics methods. Our simulation results demonstrate novel phase transitions and critical phenomena in many-body systems."
    }
    
    # Sample text selector
    st.markdown("#### 📚 Or choose a sample:")
    selected_sample = st.selectbox("Sample texts:", ["Custom"] + list(sample_texts.keys()))
    
    if selected_sample != "Custom":
        default_text = sample_texts[selected_sample]
    else:
        default_text = ""
    
    # Text input area
    user_text = st.text_area(
        "Abstract text:",
        value=default_text,
        height=150,
        placeholder="Enter a scientific abstract here..."
    )
    
    if st.button("🔍 Classify Text", type="primary") and user_text.strip():
        with st.spinner("Analyzing text..."):
            # Preprocess text
            processed_text = user_text.strip().replace("\n", " ")
            processed_text = re.sub(r'[^\w\s]', "", processed_text)
            processed_text = re.sub(r'\d+', "", processed_text)
            processed_text = re.sub(r'\s+', " ", processed_text)
            processed_text = processed_text.lower()
            
            # Get appropriate vectorizer
            if best_result['Vectorizer'] == 'Bag of Words':
                vectorizer = data_dict['bow_vectorizer']
                features = vectorizer.transform([processed_text]).toarray()
            elif best_result['Vectorizer'] == 'TF-IDF':
                vectorizer = data_dict['tfidf_vectorizer']
                features = vectorizer.transform([processed_text]).toarray()
            else:  # Embeddings
                vectorizer = data_dict['embedding_vectorizer']
                features = np.array(vectorizer.transform([processed_text]))
            
            # Make prediction
            prediction = best_model.predict(features)[0]
            predicted_class = data_dict['id_to_label'][prediction]
            
            # Display results
            st.markdown("### 🎯 Classification Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="success-box">
                    <h3>🏷️ Predicted Category: <strong>{predicted_class.upper()}</strong></h3>
                    <p><strong>Model:</strong> {best_result['Model']}</p>
                    <p><strong>Vectorizer:</strong> {best_result['Vectorizer']}</p>
                    <p><strong>Model Accuracy:</strong> {best_result['Accuracy']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Category descriptions
                category_descriptions = {
                    'astro-ph': '🌌 Astrophysics',
                    'cond-mat': '🔬 Condensed Matter',
                    'cs': '💻 Computer Science',
                    'math': '📐 Mathematics',
                    'physics': '⚛️ Physics'
                }
                
                st.markdown("#### 📋 All Categories")
                for cat_id, desc in category_descriptions.items():
                    if cat_id == predicted_class:
                        st.markdown(f"**→ {desc}** ✅")
                    else:
                        st.markdown(f"   {desc}")
            
            # Show processed text
            with st.expander("🔍 View Processed Text"):
                st.text(processed_text)

if __name__ == "__main__":
    main()