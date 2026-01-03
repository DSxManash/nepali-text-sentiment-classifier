from flask import Flask, render_template, request, jsonify
import numpy as np
import re
import joblib
import fasttext
import time
import os
import logging
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration from environment
DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
MAX_WORDS = 100

# Model configuration
FASTTEXT_REPO = "DSxManash/nepali-fasttext-cc"
FASTTEXT_FILENAME = "cc.ne.300-001.bin"

ft_model = None
svm_model = None

def download_fasttext_model():
    """Download FastText model from HuggingFace Hub if not exists locally."""
    local_path = FASTTEXT_FILENAME
    
    if os.path.exists(local_path):
        logger.info(f"FastText model found locally: {local_path}")
        return local_path
    
    logger.info(f"Downloading FastText model from HuggingFace Hub...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=FASTTEXT_REPO,
            filename=FASTTEXT_FILENAME
        )
        logger.info(f"FastText model downloaded: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Failed to download FastText model: {e}")
        return None

def load_models():
    global ft_model, svm_model
    
    svm_path = "svm_nepali_sentiment_model.pkl"
    
    # Check SVM model
    if not os.path.exists(svm_path):
        logger.error(f"SVM model not found: {svm_path}")
        return False
    
    # Download FastText if needed
    fasttext_path = download_fasttext_model()
    if not fasttext_path:
        return False
    
    try:
        start_time = time.time()
        logger.info("Loading FastText model...")
        ft_model = fasttext.load_model(fasttext_path)
        logger.info(f"FastText loaded in {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        logger.info("Loading SVM model...")
        svm_model = joblib.load(svm_path)
        logger.info(f"SVM loaded in {time.time() - start_time:.2f}s")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

models_loaded = load_models()
if not models_loaded:
    logger.warning("Application starting without models. Sentiment analysis will not work.")  

def preprocess_text(text):
    text = re.sub(r"[^\u0900-\u097F\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fasttext_vectorize(text, model):
    words = text.split()
    if not words:
        return np.zeros(model.get_dimension())
    # FastText handles OOV words via subword embeddings, no need to filter
    vectors = [model.get_word_vector(word) for word in words]
    return np.mean(vectors, axis=0)

def get_confidence_score(model, vector):
    decision = model.decision_function(vector)[0]
    confidence = 1 / (1 + np.exp(-abs(decision)))
    return round(confidence * 100, 1)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    confidence = None
    text = None
    alert = None
    
    if request.method == 'POST':
        text = request.form.get('text', '')
        
        if not ft_model or not svm_model:
            alert = "Model not loaded."
        elif len(text.split()) > MAX_WORDS:
            alert = f"Text too long. Maximum {MAX_WORDS} words allowed."
            text = ' '.join(text.split()[:MAX_WORDS])
        elif not re.search(r'[\u0900-\u097F]', text):
            alert = "Please write in Nepali."
        else:
            clean_text = preprocess_text(text)
            vector = fasttext_vectorize(clean_text, ft_model).reshape(1, -1)
            prediction = svm_model.predict(vector)[0]
            confidence = get_confidence_score(svm_model, vector)
            sentiment = 'Negative' if prediction == 0 else 'Positive'
    
    response = app.make_response(render_template("home.html", 
        sentiment=sentiment, 
        text=text, 
        confidence=confidence, 
        alert=alert))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/health', methods=['GET'])
def health():
    status = "healthy" if ft_model and svm_model else "unhealthy"
    return jsonify({"status": status})

if __name__ == "__main__":
    app.run(debug=DEBUG_MODE, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
