# Ponzi Scheme Detection

This program processes text descriptions of investment schemes to determine if they are **Ponzi schemes**. It extracts financial and textual features with NLP, computes a risk score with a Neural Network, and trains a Random Forest classifier to label schemes as `Ponzi`, `Likely Ponzi`, or `Not Ponzi`.

<br>

---

## Project Structure

| File/Folder                         | Description                                         |
|-------------------------------------|-----------------------------------------------------|
| `train.py`                          | Script to train the classifier or run inference on a single scheme file. |
| `models/`                           | Contains `tf_return_risk_model.keras` and `tf_return_risk_scaler.pkl`.    |
| `data/`                             | Contains `.txt` files of scheme descriptions, labeled by filename.       |
| `ponzi_scheme_classifier.pkl`       | Trained Random Forest classifier.                  |


## Dependencies

- **Python** 3.10 or higher  
- **Libraries**: flask, pandas, numpy, scikit-learn, tensorflow, torch, joblib, nltk, spacy, transformers, sentence-transformers,

<br>

---

## Installation

```bash
git clone https://github.com/Vijay-31-08-2005/ponzi-scheme-detection.git
cd ponzi-scheme-detection
pip install -r requirements.txt
```

### Setup Commands

```bash
python -m spacy download en_core_web_sm
```

```python
import nltk
nltk.download('vader_lexicon')
```

<br>

## Usage

The model is already trained, check [Running](https://github.com/Vijay-31-08-2005/ponzi-scheme-detection/tree/main?tab=readme-ov-file#running)

### 1. Train the Return Risk Neural Network

```bash
python models/return_risk.py
```
Trains the neural network using `return_risk_training_data.csv` and saves:
- `tf_return_risk_model.keras`
- `tf_return_risk_scaler.pkl`


### 2. Train the Ponzi Scheme Classifier

```bash
python train.py
```
Processes all `.txt` files in `data/`, computes risk scores, trains the classifier, and saves `ponzi_scheme_classifier.pkl`.

<br>


## Running


### Start the Web Application

```bash
python run.py
```
Then open http://127.0.0.1:5000/ in your browser.

### How to Use

1. **Enter a scheme description** in the text box.
2. **Click "Analyze Scheme"** to run the AI.
3. **The results will be displayed below.**
4. **Click "Show More"** to see full details.

### Web App Demo

<p align="left">
  <img src="https://res.cloudinary.com/duff0nokr/image/upload/v1747500171/ponziwebappdemo_srxwly.gif" alt="Web App Demo" width="600"/>
</p>

_This GIF demonstrates how to enter a scheme, analyze it, and view the full breakdown._

---

### Run Inference on a Single Scheme

```bash
python train.py path/to/scheme.txt
```

#### Sample Output

```bash
Parameters:
company_name: Skyline Development Fund
promised_return_percent: 4.0
return_frequency_days: 30
time_to_roi_days: 500
minimum_deposit_usd: 10000
referral_pressure: 1
whitepaper_available: 1
team_members: 1
sentiment_score: 0.6705
scam_keyword_density: 0.0
crypto_only: 0

Risk Score: 0.027123991

Classification: Not Ponzi

Probabilities:
Not Ponzi: 0.0472
Likely Ponzi: 0.9459
Ponzi: 0.0068

{'company_name': 'Skyline Development Fund', 'risk_score': 0.027123991, 'classification': 'Not Ponzi', 'probabilities': {'Not Ponzi': 0.047243107769423566, 'Likely Ponzi': 0.9459273182957394, 'Ponzi': 0.006829573934837093}}
```

---
