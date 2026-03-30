import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import re

# Configuration
DATASET_PATH = '../dataset.csv'
MODELS_DIR = '../trained_models'
SAMPLE_SIZE = 10000  # Number of rows to keep for quick search
OUTPUT_NN_MODEL = os.path.join(MODELS_DIR, 'search_nn_model.pkl')
OUTPUT_TFIDF_MODEL = os.path.join(MODELS_DIR, 'search_tfidf_vectorizer.pkl')
OUTPUT_CORPUS = os.path.join(MODELS_DIR, 'search_corpus.csv')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return " ".join(text.split())

def main():
    print("🚀 Initializing ML Search Training Pipeline...")
    
    # Ensure models directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # 1. Load Data
    print(f"Loading '{DATASET_PATH}'...")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: '{DATASET_PATH}' not found!")
        return

    # To save memory and speed, we read a randomized sample. We can just read the whole file 
    # and sample, or read a subset chunk. Given it's 165MB, pandas can handle it for training
    df = pd.read_csv(DATASET_PATH)
    
    print(f"Total rows in dataset: {len(df)}")
    
    # Drop rows without necessary fields
    df = df.dropna(subset=['job_title', 'job_description', 'all_skills'])
    
    # Using the full dataset as requested
    # Removing the sampling limit
    df = df.reset_index(drop=True)
    
    print(f"Using all {len(df)} rows for the search index.")

    # 2. Preprocess strings
    print("Vectorizing Text Data (TF-IDF)...")
    
    # Combine relevant columns into a single search document representation
    # Weighting: We can just concatenate them. If we repeat job_title, it gets higher TF-IDF weight
    df['search_text'] = (
        df['job_title'].astype(str) + " " + 
        df['job_title'].astype(str) + " " +  # Repeated for weight
        df['all_skills'].astype(str) + " " + 
        df['company'].astype(str) + " " +
        df['job_description'].astype(str)
    ).apply(clean_text)

    # 3. Fit TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['search_text'])

    # 4. Train NearestNeighbors Model
    print("Training K-Nearest Neighbors...")
    nn_model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix)

    # 5. Serialize and Save
    print("Saving models and lightweight corpus to disk...")
    
    joblib.dump(tfidf, OUTPUT_TFIDF_MODEL)
    joblib.dump(nn_model, OUTPUT_NN_MODEL)
    
    # Save the dataframe corpus (Drop search text to save space)
    df_corpus = df.drop(columns=['search_text'])
    df_corpus.to_csv(OUTPUT_CORPUS, index=False)
    
    print(f"✅ Training successful!")
    print(f" - TF-IDF Vectorizer saved to: {OUTPUT_TFIDF_MODEL}")
    print(f" - NN Model saved to: {OUTPUT_NN_MODEL}")
    print(f" - Search Corpus saved to: {OUTPUT_CORPUS}")

if __name__ == "__main__":
    # Change to current directory first
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    main()
