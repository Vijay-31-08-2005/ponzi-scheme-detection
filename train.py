import os
import pandas as pd
import numpy as np
import re
import joblib
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


# Load the return risk model and decision tree scaler
risk_model = load_model("tf_return_risk_model.keras")
scaler = joblib.load("tf_return_risk_scaler.pkl")

# Mapping from filename to class label
label_mapping = {
    "NotPonzi": "Not Ponzi",
    "LikelyPonzi": "Likely Ponzi",
    "Ponzi": "Ponzi"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load spaCy model with GPU support (if available)
nlp = spacy.load("en_core_web_sm")  # Use GPU 0
if spacy.prefer_gpu():
    nlp.to_gpu(0) 

# Load HuggingFace QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=0)  # Use GPU 0

# Load SentenceTransformer model (with explicit device handling)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)


# Semantic scam keywords
scam_phrases = [
    "guaranteed return", "double your investment", "limited time only", 
    "risk-free", "act fast", "instant profit", "earn while you sleep"
]

def parse_scheme_text(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    doc = nlp(text)

    # Company name (NER or QA)
    company_name = next((ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]), "Unknown")
    if company_name == "Unknown":
        company_name = qa_pipeline(question="What is the name of the company?", context=text)['answer']

    # Promised return (QA)
    try:
        return_text = qa_pipeline(question="What is the promised return?", context=text)['answer']
        promised_return_percent = float(re.search(r'\d+', return_text).group())
    except:
        promised_return_percent = 0.0

    # Return frequency (QA/rule-based)
    return_frequency_days = 30 if "monthly" in text.lower() else 7 if "weekly" in text.lower() else 0

    # Time to ROI
    try:
        roi_text = qa_pipeline(question="In how many days do I get a return?", context=text)['answer']
        time_to_roi_days = int(re.search(r'\d+', roi_text).group())
    except:
        time_to_roi_days = 0

    # Minimum deposit
    try:
        deposit_text = qa_pipeline(question="What is the minimum deposit?", context=text)['answer']
        minimum_deposit_usd = int(re.sub(r"[^\d]", "", deposit_text))
    except:
        minimum_deposit_usd = 0

    # Referral, whitepaper, team
    referral_pressure = 1 if "referral" in text.lower() else 0
    whitepaper_available = 1 if "whitepaper" in text.lower() else 0
    team_members = 1 if re.search(r"(CEO|Head of Ops|founder|team)", text, re.IGNORECASE) else 0

    # Sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']

    # Scam keyword detection (semantic similarity)
    sentences = text.split('.')
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    keyword_embeddings = embedder.encode(scam_phrases, convert_to_tensor=True)

    scam_scores = [util.pytorch_cos_sim(sentence_embeddings, ke).max().item() for ke in keyword_embeddings]
    scam_keyword_density = sum(score > 0.6 for score in scam_scores) / max(len(sentences), 1)

    # Crypto only
    crypto_only = 1 if 'crypto' in text.lower() else 0

    return {
        "company_name": company_name,
        "promised_return_percent": promised_return_percent,
        "return_frequency_days": return_frequency_days,
        "time_to_roi_days": time_to_roi_days,
        "minimum_deposit_usd": minimum_deposit_usd,
        "referral_pressure": referral_pressure,
        "whitepaper_available": whitepaper_available,
        "team_members": team_members,
        "sentiment_score": sentiment_score,
        "scam_keyword_density": scam_keyword_density,
        "crypto_only": crypto_only
    }


# Function to calculate return risk score
def calculate_return_risk(params):
    features = np.array([[
        params["promised_return_percent"],
        params["return_frequency_days"],
        params["time_to_roi_days"],
        params["minimum_deposit_usd"]
    ]])
    features_scaled = scaler.transform(features)
    return risk_model.predict(features_scaled)[0][0]

# Function to train the decision tree model

def train_decision_tree(folder_path):
    data = []
    labels = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            params = parse_scheme_text(file_path)
            risk_score = calculate_return_risk(params)

            label = "Unknown"
            for key in label_mapping:
                if file_name.startswith(key):
                    label = label_mapping[key]
                    break
                
            data.append([
                params["promised_return_percent"],
                params["return_frequency_days"],
                params["time_to_roi_days"],
                params["minimum_deposit_usd"],
                params["referral_pressure"],
                params["whitepaper_available"],
                params["team_members"],
                params["sentiment_score"],
                params["scam_keyword_density"],
                params["crypto_only"],
                risk_score
            ])
            labels.append(label)
    
    df = pd.DataFrame(data, columns=[
        "promised_return_percent", "return_frequency_days", "time_to_roi_days",
        "minimum_deposit_usd", "referral_pressure", "whitepaper_available",
        "team_members", "sentiment_score", "scam_keyword_density", "crypto_only", "risk_score"
    ])
    
    X = df.drop(columns=["risk_score"])
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter distribution for Randomized Search
    param_dist = {
        "n_estimators": randint(100, 300),
        "max_depth": randint(5, 30),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    }

    base_clf = RandomForestClassifier(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    best_clf = random_search.best_estimator_
    print("Best Hyperparameters:", random_search.best_params_)

    y_pred = best_clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(best_clf, "ponzi_scheme_classifier.pkl")
    print("Optimized decision tree model saved.")

# Function to infer from a single scheme text file
def infer_scheme(file_path):
    params = parse_scheme_text(file_path)
    risk_score = calculate_return_risk(params)
    clf = joblib.load("ponzi_scheme_classifier.pkl")
    
    features = np.array([[
        params["promised_return_percent"],
        params["return_frequency_days"],
        params["time_to_roi_days"],
        params["minimum_deposit_usd"],
        params["referral_pressure"],
        params["whitepaper_available"],
        params["team_members"],
        params["sentiment_score"],
        params["scam_keyword_density"],
        params["crypto_only"]
    ]])
    
    # Get prediction and probabilities
    prediction = clf.predict(features)
    probabilities = clf.predict_proba(features)

    # Create a dictionary to hold the results
    result = {
        "company_name": params["company_name"],
        "risk_score": risk_score,
        "classification": prediction[0],
        "probabilities": {}
    }

    # Check the size of probabilities and assign values accordingly
    if probabilities.shape[1] > 1:
        result["probabilities"] = {
            "Not Ponzi": probabilities[0][0],
            "Likely Ponzi": probabilities[0][1],
            "Ponzi": probabilities[0][2]
        }
    else:
        result["probabilities"] = {
            "Not Ponzi": 1.0,  # Assuming the only class predicted
            "Likely Ponzi": 0.0,
            "Ponzi": 0.0
        }

    # Print all parameters and scores
    print("Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    print("\nRisk Score:", risk_score)
    print("\nClassification:", prediction[0])
    print("\nProbabilities:")
    for class_name, prob in result["probabilities"].items():
        print(f"{class_name}: {prob:.4f}")

    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        scheme_file = sys.argv[1]
        result = infer_scheme(scheme_file)
        print(result)
    else:
        folder_path = "data"  # Folder containing all .txt files
        train_decision_tree(folder_path)