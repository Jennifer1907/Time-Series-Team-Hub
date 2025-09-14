import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import html
from collections import Counter, defaultdict
from typing import List, Dict, Literal, Union
import warnings
warnings.filterwarnings('ignore')

# Streamlit specific imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.validation import check_X_y, check_array
    from sklearn.utils.multiclass import unique_labels
    from scipy.spatial.distance import cdist
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Scientific Abstract Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
CATEGORY_NAMES = {
    'astro-ph': 'Astrophysics',
    'cond-mat': 'Condensed Matter',
    'cs': 'Computer Science',
    'math': 'Mathematics',
    'physics': 'Physics'
}

# Custom KNN Classifier
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

    def _calculate_saliency_weights(self, X_test):
        feature_vars = np.var(self.X_train_, axis=0)
        feature_vars = np.where(feature_vars == 0, 1e-10, feature_vars)
        saliency = 1.0 / (1.0 + feature_vars)
        
        saliency_weights = []
        for x_test in X_test:
            weighted_distance = np.sum(saliency * (x_test - np.mean(self.X_train_, axis=0))**2)
            saliency_weights.append(1.0 / (1.0 + weighted_distance))
        return np.array(saliency_weights)

# Embedding Vectorizer
class EmbeddingVectorizer:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self.model = None

    @st.cache_resource
    def load_model(_self):
        """Load the sentence transformer model with caching"""
        try:
            _self.model = SentenceTransformer(_self.model_name)
            return _self.model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    def _format_inputs(self, texts: List[str], mode: Literal["query", "passage"]):
        if mode not in {"query", "passage"}:
            raise ValueError("Mode must be either 'query' or 'passage'")
        return [f"{mode}: {t.strip()}" for t in texts]

    def transform(self, texts: List[str], mode: Literal["query", "passage"] = "query"):
        if self.model is None:
            self.model = self.load_model()
        
        if self.model is None:
            return np.random.random((len(texts), 384))  # Fallback random embeddings
        
        if mode == "raw":
            inputs = texts
        else:
            inputs = self._format_inputs(texts, mode)
        
        embeddings = self.model.encode(inputs, normalize_embeddings=self.normalize)
        return embeddings.tolist()

# Sample data generation for demo purposes
@st.cache_data
def generate_sample_data():
    """Generate sample training data for demo"""
    # Sample abstracts for each category
    sample_abstracts = {
        'astro-ph': [
            "we study the formation of stars in distant galaxies using hubble space telescope observations stellar evolution galactic structure cosmic radiation",
            "black hole accretion disk matter spiral galaxy dark matter cosmic microwave background radiation stellar formation",
            "supernova explosion neutron star pulsar binary system gravitational waves cosmic ray detection stellar remnant",
            "galaxy cluster dark energy cosmic expansion hubble constant redshift measurement astronomical observation cosmological parameter",
            "exoplanet detection planetary system habitable zone stellar activity radial velocity transit photometry atmospheric composition"
        ],
        'cond-mat': [
            "we investigate the electronic properties of graphene using density functional theory calculations band structure conductivity",
            "superconducting material high temperature critical temperature cooper pair electron phonon interaction magnetic field",
            "quantum phase transition spin glass frustrated magnet monte carlo simulation statistical mechanics thermodynamic property",
            "semiconductor device heterostructure quantum well electron mobility carrier concentration optical property",
            "magnetic material ferromagnetism antiferromagnetism spin orbit coupling exchange interaction magnetic moment"
        ],
        'cs': [
            "machine learning algorithm neural network deep learning training data classification regression optimization",
            "computer vision image processing object detection convolutional neural network feature extraction pattern recognition",
            "natural language processing text mining sentiment analysis word embedding language model transformer architecture",
            "algorithm complexity computational time space efficiency optimization problem graph theory data structure",
            "software engineering system design distributed computing cloud computing scalability performance evaluation"
        ],
        'math': [
            "we prove a new theorem in algebraic geometry using cohomology theory manifold differential equation",
            "numerical analysis finite element method partial differential equation convergence stability error estimation",
            "probability theory stochastic process random variable distribution function limit theorem statistical inference",
            "topology geometric structure homeomorphism continuous function metric space topological property",
            "number theory prime number integer sequence diophantine equation modular form arithmetic function"
        ],
        'physics': [
            "quantum mechanics wave function schrödinger equation measurement uncertainty principle entanglement superposition",
            "particle physics standard model elementary particle accelerator collision high energy interaction",
            "condensed matter physics solid state electron band theory lattice vibration thermal property",
            "statistical mechanics thermodynamics entropy temperature equilibrium phase transition critical phenomenon",
            "optics laser light electromagnetic radiation nonlinear optics photonic crystal optical fiber"
        ]
    }
    
    # Create training data
    X_train = []
    y_train = []
    
    for i, (category, abstracts) in enumerate(sample_abstracts.items()):
        for abstract in abstracts:
            X_train.append(abstract)
            y_train.append(i)
    
    return X_train, y_train

# Text preprocessing
def preprocess_text(text):
    """Preprocess text similar to training data"""
    text = text.strip().replace("\n", " ")
    text = re.sub(r'[^\w\s]', "", text)
    text = re.sub(r'\d+', "", text)
    text = re.sub(r'\s+', " ", text)
    return text.lower()

# Saliency computation
def compute_word_saliency(text, model, vectorizer, original_prediction, class_names):
    """Compute saliency scores for each word in the text"""
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) <= 1:
        return words, np.array([0.5])
    
    # Get original prediction confidence
    original_embedding = vectorizer.transform([text])
    
    saliencies = []
    
    for i, word in enumerate(words):
        # Create masked text by removing the current word
        masked_words = words.copy()
        masked_words.pop(i)
        masked_text = ' '.join(masked_words)
        
        if not masked_text.strip():
            saliencies.append(0.0)
            continue
        
        try:
            masked_embedding = vectorizer.transform([masked_text])
            masked_prediction = model.predict(masked_embedding)[0]
            
            
            if masked_prediction != original_prediction:
                saliencies.append(1.0)
            else:
                saliencies.append(0.3)
        except:
            saliencies.append(0.1)
    
    # Normalize saliencies
    saliencies = np.array(saliencies)
    if np.ptp(saliencies) > 1e-6:
    # min-max chuẩn
        saliencies = (saliencies - saliencies.min()) / (rng + 1e-8)
    else:
    # ---- Cách A: z-score + sigmoid để kéo giãn nhỏ ----
        m, std = saliencies.mean(), saliencies.std()
        if std > 0:
            z = (saliencies - m) / (std + 1e-8)
            g = 5.0  # hệ số "gain" (tăng nếu muốn tương phản mạnh hơn)
            s = 1.0 / (1.0 + np.exp(-g * z))          # sigmoid -> (0,1)
            saliencies = (s - s.min()) / (s.max() - s.min() + 1e-8)
        else:
        # ---- Cách B: rank-based (mọi giá trị bằng nhau) ----
            idx = np.arange(len(saliencies))
            ranks = np.argsort(np.argsort(idx)).astype(float)
            saliencies = ranks / max(len(saliencies) - 1, 1)
    
    return words, saliencies

# Saliency visualization
def create_saliency_html(words, saliencies, predicted_class):
    """Create HTML visualization for word saliency"""
    if not words or len(words) != len(saliencies):
        return "<p>No words to visualize.</p>"
    
    # Color mapping
    def get_color(saliency):
        # Blue scale: light blue to dark blue
        intensity = min(max(saliency, 0), 1)
        r = int(255 - intensity * 200)
        g = int(255 - intensity * 150)
        b = 255
        return f"rgb({r},{g},{b})"
    
    # Create HTML
    html_parts = []
    html_parts.append(f"<div style='font-family: Arial, sans-serif; line-height: 1.8;'>")
    html_parts.append(f"<p><strong>Predicted Class:</strong> {predicted_class}</p>")
    html_parts.append(f"<div style='margin-bottom: 10px;'>")
    
    for word, saliency in zip(words, saliencies):
        color = get_color(saliency)
        html_parts.append(
            f"<span style='background-color: {color}; padding: 2px 4px; "
            f"margin: 1px; border-radius: 3px; display: inline-block;' "
            f"title='Saliency: {saliency:.3f}'>{html.escape(word)}</span> "
        )
    
    html_parts.append("</div></div>")
    return "".join(html_parts)

# Main app
def main():
    st.title("🔬 Scientific Abstract Classifier")
    st.markdown("*Classify scientific abstracts into categories: Astrophysics, Condensed Matter, Computer Science, Mathematics, Physics*")
    
    # Sidebar
    st.sidebar.header("Settings")
    vectorizer_type = st.sidebar.selectbox(
        "Choose Vectorizer",
        ["Embeddings", "TF-IDF", "Bag of Words"]
    )
    
    model_type = st.sidebar.selectbox(
        "Choose Model",
        ["Custom KNN (α=0.5)", "Custom KNN (α=0.7)", "KNN Weighted", "Naive Bayes", "Decision Tree"]
    )
    
    show_saliency = st.sidebar.checkbox("Show Word Saliency Analysis", value=True)
    
    # Load or generate data
    with st.spinner("Loading models and data..."):
        X_train, y_train = generate_sample_data()
        
        # Initialize vectorizers
        if vectorizer_type == "Embeddings":
            vectorizer = EmbeddingVectorizer()
            X_train_vec = np.array(vectorizer.transform(X_train))
        elif vectorizer_type == "TF-IDF":
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X_train_vec = vectorizer.fit_transform(X_train).toarray()
        else:  # Bag of Words
            vectorizer = CountVectorizer(max_features=1000, stop_words='english')
            X_train_vec = vectorizer.fit_transform(X_train).toarray()
        
        # Initialize model
        if "Custom KNN (α=0.5)" in model_type:
            model = CustomKNN(n_neighbors=5, voting='custom', alpha=0.5)
        elif "Custom KNN (α=0.7)" in model_type:
            model = CustomKNN(n_neighbors=5, voting='custom', alpha=0.7)
        elif "KNN Weighted" in model_type:
            model = CustomKNN(n_neighbors=5, voting='weighted')
        elif "Naive Bayes" in model_type:
            model = GaussianNB()
        else:  # Decision Tree
            model = DecisionTreeClassifier(random_state=42, max_depth=10)
        
        # Train model
        model.fit(X_train_vec, y_train)
    
    st.success("✅ Models loaded successfully!")
    
    # Input section
    st.header("📝 Enter Abstract")
    
    # Example abstracts
    examples = {
        "Astrophysics": "We present observations of a supermassive black hole at the center of a nearby galaxy using the Hubble Space Telescope. Our analysis reveals strong evidence for relativistic jets and accretion disk formation around the black hole.",
        "Computer Science": "We propose a novel deep learning architecture for natural language processing tasks. Our model combines transformer attention mechanisms with convolutional layers to achieve state-of-the-art performance on sentiment analysis and text classification.",
        "Mathematics": "We prove a new theorem in algebraic topology concerning the homology groups of certain fiber bundles. The proof uses spectral sequence techniques and provides insights into the geometric structure of these spaces.",
        "Physics": "We study quantum entanglement in a two-photon system using polarization measurements. Our experimental results demonstrate violation of Bell's inequality and confirm the non-local nature of quantum mechanics.",
        "Condensed Matter": "We investigate the electronic properties of graphene nanoribbons using density functional theory calculations. Our results show how edge effects and quantum confinement lead to novel electronic band structures."
    }
    
    # Example selector
    selected_example = st.selectbox("Choose an example:", ["Custom"] + list(examples.keys()))
    
    if selected_example != "Custom":
        default_text = examples[selected_example]
    else:
        default_text = ""
    
    # Text input
    user_input = st.text_area(
        "Paste your scientific abstract here:",
        value=default_text,
        height=150,
        placeholder="Enter the scientific abstract you want to classify..."
    )
    
    if st.button("🔍 Classify Abstract", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing abstract..."):
                # Preprocess text
                processed_text = preprocess_text(user_input)
                
                # Vectorize
                if vectorizer_type == "Embeddings":
                    test_vec = np.array(vectorizer.transform([processed_text]))
                else:
                    test_vec = vectorizer.transform([processed_text]).toarray()
                
                # Predict
                prediction = model.predict(test_vec)[0]
                predicted_category = CATEGORIES_TO_SELECT[prediction]
                predicted_name = CATEGORY_NAMES[predicted_category]
                
                # Get prediction probabilities if possible
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(test_vec)[0]
                    else:
                        # For KNN, create dummy probabilities
                        proba = np.zeros(len(CATEGORIES_TO_SELECT))
                        proba[prediction] = 0.85
                        # Distribute remaining probability
                        remaining = 0.15
                        for i in range(len(proba)):
                            if i != prediction:
                                proba[i] = remaining / (len(proba) - 1)
                except:
                    proba = np.zeros(len(CATEGORIES_TO_SELECT))
                    proba[prediction] = 1.0
                
                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.header("🎯 Prediction Results")
                    st.success(f"**Predicted Category:** {predicted_name}")
                    st.info(f"**Confidence:** {proba[prediction]:.1%}")
                    
                    # Show all probabilities
                    st.subheader("📊 All Categories")
                    prob_df = pd.DataFrame({
                        'Category': [CATEGORY_NAMES[cat] for cat in CATEGORIES_TO_SELECT],
                        'Probability': proba
                    }).sort_values('Probability', ascending=False)
                    
                    st.dataframe(prob_df, use_container_width=True)
                
                with col2:
                    st.header("📈 Probability Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar([CATEGORY_NAMES[cat] for cat in CATEGORIES_TO_SELECT], proba)
                    ax.set_ylabel('Probability')
                    ax.set_title('Category Probabilities')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Highlight the predicted category
                    bars[prediction].set_color('orange')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Saliency analysis
                if show_saliency and vectorizer_type == "Embeddings":
                    st.header("🔍 Word Saliency Analysis")
                    st.markdown("*Words are colored by their importance in the prediction. Darker blue = more important.*")
                    
                    try:
                        words, saliencies = compute_word_saliency(
                            processed_text, model, vectorizer, prediction, CATEGORIES_TO_SELECT
                        )
                        
                        # Create and display HTML visualization
                        html_viz = create_saliency_html(words, saliencies, predicted_name)
                        st.components.v1.html(html_viz, height=200)
                        
                        # Show top contributing words (filtered)
                        st.subheader("🔝 Top Contributing Words")
                        
                        # Filter out stop words and short words
                        filtered_word_saliency = []
                        EXTENDED_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
                            'we', 'our', 'us', 'using', 'used', 'show', 'study', 'present', 
                            'investigate', 'analyze', 'result', 'method', 'paper', 'work',
                            'new', 'novel', 'propose', 'develop', 'based', 'obtained'
                        })
                        for word, saliency in zip(words, saliencies):
                            if (word.lower() not in EXTENDED_STOP_WORDS and 
                                len(word) >= 3 and 
                                word.isalpha() and
                                saliency > 0.1):  # Only include words with meaningful saliency
                                filtered_word_saliency.append((word, saliency))
                        
                        # Sort by saliency score
                        filtered_word_saliency.sort(key=lambda x: x[1], reverse=True)
                        
                        if filtered_word_saliency:
                            # Take top 10 meaningful words
                            top_words_df = pd.DataFrame(
                                filtered_word_saliency[:10], 
                                columns=['Word', 'Saliency Score']
                            )
                            st.dataframe(top_words_df, use_container_width=True)
                            
                            # Show statistics
                            st.caption(f"Showing {len(top_words_df)} most important words (stop words filtered out)")
                        else:
                            st.warning("No significant contributing words found after filtering stop words.")
                        
                    except Exception as e:
                        st.warning(f"Could not compute saliency analysis: {str(e)}")
        else:
            st.warning("Please enter an abstract to classify.")
    
    # Information section
    st.header("ℹ️ About")
    with st.expander("Model Information"):
        st.markdown("""
        **Categories:**
        - **Astrophysics (astro-ph):** Astronomy, cosmology, stellar physics
        - **Condensed Matter (cond-mat):** Materials science, solid state physics
        - **Computer Science (cs):** Algorithms, machine learning, software engineering
        - **Mathematics (math):** Pure and applied mathematics
        - **Physics (physics):** General physics, quantum mechanics, particle physics
        
        **Features:**
        - Multiple vectorization methods (Embeddings, TF-IDF, Bag of Words)
        - Custom KNN classifier with different voting schemes
        - Word saliency analysis for interpretability
        - Real-time classification
        """)
    
    with st.expander("How to Use"):
        st.markdown("""
        1. **Choose Settings:** Select your preferred vectorizer and model in the sidebar
        2. **Input Text:** Enter or select an example scientific abstract
        3. **Classify:** Click the "Classify Abstract" button
        4. **Analyze Results:** View the prediction, confidence scores, and word importance
        
        **Tips:**
        - Longer, more detailed abstracts typically give better results
        - The word saliency analysis helps understand which terms influenced the prediction
        - Try different model combinations to see how they perform
        """)

if __name__ == "__main__":
    main()