import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Import your training modules
from spam_model import SpamClassifier
from utils import load_model_artifacts, save_model_artifacts

# Configure Streamlit page
st.set_page_config(
    page_title="🛡️ Spam Slayer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .spam-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    .ham-box {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .saliency-word {
        display: inline-block;
        margin: 2px;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = 0
    if 'training_status' not in st.session_state:
        st.session_state.training_status = ""

def render_saliency_heatmap(tokens, saliency_scores):
    """Render saliency heatmap using Plotly"""
    if not tokens or not saliency_scores:
        return None
    
    # Normalize saliency scores to 0-1 range
    if len(saliency_scores) > 1:
        min_score = min(saliency_scores)
        max_score = max(saliency_scores)
        if max_score > min_score:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in saliency_scores]
        else:
            normalized_scores = [0.5] * len(saliency_scores)
    else:
        normalized_scores = [0.5] * len(saliency_scores)
    
    # Create HTML with colored spans
    html_content = ""
    for token, score in zip(tokens, normalized_scores):
        # Color intensity based on saliency score
        red_intensity = int(255 * score)
        color = f"rgba(255, {255 - red_intensity}, {255 - red_intensity}, 0.7)"
        html_content += f'<span class="saliency-word" style="background-color: {color};">{token}</span> '
    
    return html_content

def load_trained_model():
    """Load pre-trained model artifacts if they exist"""
    model_files = [
        'model_artifacts.pkl',
        'faiss_index.bin',
        'train_metadata.json',
        'class_weights.json',
        'model_config.json'
    ]
    
    if all(os.path.exists(f) for f in model_files):
        try:
            st.session_state.classifier = SpamClassifier.load_from_files()
            st.session_state.model_trained = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    return False

def train_model_callback(language):
    """Callback function for model training"""
    try:
        # Initialize classifier
        classifier = SpamClassifier()
        
        # Update progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading dataset...")
        progress_bar.progress(10)
        
        # Load data based on language selection
        source = 'kaggle' if language == 'Vietnamese' else 'gdrive'
        messages, labels = classifier.load_dataset(source=source)
        
        status_text.text(f"Loaded {len(messages)} messages. Starting training...")
        progress_bar.progress(30)
        
        # Train the model
        results = classifier.train(
            messages, labels,
            progress_callback=lambda p, msg: (
                progress_bar.progress(min(30 + int(p * 0.6), 90)),
                status_text.text(msg)
            )
        )
        
        status_text.text("Saving model artifacts...")
        progress_bar.progress(95)
        
        # Save model artifacts
        classifier.save_to_files()
        
        # Update session state
        st.session_state.classifier = classifier
        st.session_state.model_trained = True
        
        progress_bar.progress(100)
        status_text.text("Training completed successfully!")
        
        # Show training results
        st.success("✅ Model training completed!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", len(messages))
        with col2:
            st.metric("Best Alpha", f"{results['best_alpha']:.2f}")
        with col3:
            best_accuracy = max(results['accuracy_results'].values())
            st.metric("Best Accuracy", f"{best_accuracy:.1%}")
        
        return True
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Spam Slayer</h1>
        <p>Advanced Multilingual Spam Detection with Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language selection
        language = st.selectbox(
            "Select Language",
            ["Vietnamese", "English"],
            help="Vietnamese: Loads Kaggle dataset\nEnglish: Loads Google Drive dataset"
        )
        
        st.markdown("---")
        
        # Model status
        st.subheader("📊 Model Status")
        if st.session_state.model_trained:
            st.success("✅ Model Ready")
        else:
            st.warning("⏳ Model Not Trained")
        
        # Try to load existing model
        if not st.session_state.model_trained:
            if st.button("🔄 Load Existing Model"):
                with st.spinner("Loading model..."):
                    if load_trained_model():
                        st.success("Model loaded successfully!")
                        st.rerun()
                    else:
                        st.info("No pre-trained model found. Please train a new model.")
        
        st.markdown("---")
        
        # Training section
        st.subheader("🎯 Model Training")
        
        if st.button("🚀 Train New Model", disabled=False):
            st.info(f"Starting training with {language} dataset...")
            train_model_callback(language)
    
    # Main content
    if not st.session_state.model_trained:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Spam Slayer!
        
        Get started by training a model:
        
        1. **Select Language** in the sidebar (Vietnamese or English)
        2. **Click "Train New Model"** to start training
        3. **Wait for training to complete** (this may take a few minutes)
        4. **Start classifying messages!**
        
        ### 🌟 Features:
        - **Multilingual Support**: Vietnamese and English datasets
        - **Advanced AI**: Uses multilingual E5 embeddings with weighted KNN
        - **Explainable AI**: Saliency heatmaps show which words influence predictions
        - **Spam Subcategorization**: Detailed spam type classification
        - **Real-time Processing**: Instant classification results
        """)
        
        # Show training demo
        with st.expander("📖 How It Works"):
            st.markdown("""
            **Spam Slayer** uses state-of-the-art machine learning techniques:
            
            1. **Text Embedding**: Converts messages to numerical representations using multilingual E5
            2. **Similarity Search**: Uses FAISS for efficient nearest neighbor search
            3. **Weighted Classification**: Combines similarity scores with class weights and saliency
            4. **Explainability**: Computes token-level importance scores
            5. **Subcategorization**: Classifies spam into specific types (promotional, system alerts, etc.)
            """)
    
    else:
        # Classification interface
        st.markdown("## 🔍 Message Classification")
        
        # Input section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_message = st.text_area(
                "Enter message to classify:",
                placeholder="Type your message here...",
                height=100
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            classify_button = st.button("🔍 Classify", type="primary")
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                k_neighbors = st.slider("K Neighbors", 1, 20, 5)
                show_neighbors = st.checkbox("Show Similar Messages", False)
                explain_prediction = st.checkbox("Show Explainability", True)
        
        # Classification results
        if classify_button and user_message.strip():
            with st.spinner("Analyzing message..."):
                try:
                    # Get prediction
                    result = st.session_state.classifier.classify_message(
                        user_message,
                        k=k_neighbors,
                        explain=explain_prediction
                    )
                    
                    # Display prediction
                    prediction = result['prediction']
                    vote_scores = result['vote_scores']
                    
                    if prediction == 'spam':
                        st.markdown(f"""
                        <div class="prediction-box spam-box">
                            🚨 SPAM DETECTED 🚨
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show subcategory
                        if 'subcategory' in result and result['subcategory']:
                            subcategory_map = {
                                'spam_quangcao': '📢 Promotional/Advertisement',
                                'spam_hethong': '⚠️ System Alert/Phishing',
                                'spam_khac': '🔍 Other Spam Type'
                            }
                            subcategory_name = subcategory_map.get(
                                result['subcategory'], 
                                result['subcategory']
                            )
                            st.info(f"**Spam Type:** {subcategory_name}")
                    
                    else:
                        st.markdown(f"""
                        <div class="prediction-box ham-box">
                            ✅ LEGITIMATE MESSAGE ✅
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show confidence scores
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ham Score", f"{vote_scores['ham']:.3f}")
                    with col2:
                        st.metric("Spam Score", f"{vote_scores['spam']:.3f}")
                    with col3:
                        total_votes = sum(vote_scores.values())
                        if total_votes > 0:
                            confidence = max(vote_scores.values()) / total_votes
                        else:
                            confidence = 0.0
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Explainability section
                    if explain_prediction and 'tokens' in result and 'saliency_scores' in result:
                        st.markdown("### 🔬 Explainability Analysis")
                        
                        # Saliency heatmap
                        heatmap_html = render_saliency_heatmap(
                            result['tokens'], 
                            result['saliency_scores']
                        )
                        
                        if heatmap_html:
                            st.markdown("**Word Importance Heatmap:**")
                            st.markdown(f'<div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa;">{heatmap_html}</div>', unsafe_allow_html=True)
                            st.caption("Darker red indicates higher influence on spam classification")
                    
                    # Similar messages section
                    if show_neighbors and 'neighbors' in result:
                        st.markdown("### 📋 Similar Training Messages")
                        
                        for i, neighbor in enumerate(result['neighbors'][:5], 1):
                            with st.expander(f"Similar Message #{i} - {neighbor['label'].upper()}"):
                                st.write(f"**Similarity:** {neighbor['score']:.3f}")
                                st.write(f"**Weight:** {neighbor['weight']:.3f}")
                                st.write(f"**Message:** {neighbor['message']}")
                
                except Exception as e:
                    st.error(f"Classification error: {str(e)}")
        
        elif classify_button:
            st.warning("Please enter a message to classify.")
        
        # Usage statistics
        with st.expander("📊 Model Information"):
            if hasattr(st.session_state.classifier, 'model_info'):
                info = st.session_state.classifier.model_info
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Training Dataset:**", info.get('dataset_size', 'N/A'))
                    st.write("**Model Type:**", info.get('model_name', 'multilingual-e5-base'))
                with col2:
                    st.write("**Best Alpha:**", info.get('best_alpha', 'N/A'))
                    st.write("**Training Date:**", info.get('training_date', 'N/A'))

if __name__ == "__main__":
    main()