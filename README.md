---
title: NepSentiX
emoji: 💬
colorFrom: red
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# NepSentiX - Nepali Sentiment Analyzer

A machine learning-based web application for analyzing sentiment in Nepali text. The system classifies input text as **Positive** or **Negative** with a confidence score.

## Live Demo

Try the application: [huggingface.co/spaces/DSxManash/nepali-sentiment-classifier](https://huggingface.co/spaces/DSxManash/nepali-sentiment-classifier)

## Features

- **Nepali Language Support**: Designed specifically for Devanagari script
- **Binary Classification**: Positive/Negative sentiment detection
- **Confidence Score**: Probability-based confidence percentage
- **Real-time Analysis**: Instant predictions via web interface
- **Input Validation**: Handles non-Nepali text and word limits (max 100 words)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask (Python 3.9+) |
| Word Embeddings | FastText |
| Classifier | Support Vector Machine (SVM) |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Docker on Hugging Face Spaces |

## Models

### FastText Word Embeddings
- **File**: `cc.ne.300-001.bin`
- **Dimensions**: 300
- **Source**: Common Crawl + Wikipedia
- **Size**: ~3.6 GB
- **Hosted**: [huggingface.co/DSxManash/nepali-fasttext-cc](https://huggingface.co/DSxManash/nepali-fasttext-cc)

### SVM Classifier
- **File**: `svm_nepali_sentiment_model.pkl`
- **Training Data**: 28,800 labeled Nepali sentences (balanced dataset)
- **Size**: ~26 MB

## Local Setup

### Prerequisites
- Python 3.9 or higher
- ~4 GB disk space for FastText model

### Installation

```bash
# Clone repository
git clone https://github.com/DSxManash/nepali-text-sentiment-classifier.git
cd nepali-text-sentiment-classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate 

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

The FastText model will be automatically downloaded from Hugging Face Hub on first run.

## Usage

1. Enter Nepali text in the input box (maximum 100 words)
2. Click **Analyze** or press `Ctrl + Enter`
3. View the sentiment result with confidence score

## Project Structure

```
nepali-text-sentiment-classifier/
├── app.py                      
├── requirements.txt         
├── Dockerfile                  
├── README.md                  
├── templates/
│   └── home.html             
├── static/
│   └── style.css              
├── resources/
│   ├── nepali_sentiment_classification.ipynb  
│   └── balanced_nepali_sentiment_dataset_28800.csv
└── svm_nepali_sentiment_model.pkl  
```

## Deployment

The application is deployed on **Hugging Face Spaces** using Docker:

- FastText model is downloaded at runtime from Hugging Face Hub
- SVM model is bundled in the Docker image
- Application runs on port 7860

## How It Works

1. **Input Processing**: Text is cleaned by removing non-Devanagari characters
2. **Vectorization**: FastText generates 300-dimensional word embeddings
3. **Classification**: SVM predicts sentiment based on averaged word vectors
4. **Confidence**: Sigmoid function converts SVM decision score to probability

## Author

**DSxManash**
- Hugging Face: [@DSxManash](https://huggingface.co/DSxManash)
- GitHub: [@DSxManash](https://github.com/DSxManash)
