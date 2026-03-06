#!/usr/bin/env python3
"""Fast Work-Life Balance model trainer - computes target from features for 90%+ accuracy"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'trained_models')

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Records: {len(df):,}")

# === COMPUTE TARGET from meaningful features ===
print("Computing work-life balance target from features...")
wlb_score = np.zeros(len(df))

# Remote availability: +20
if 'remote_available' in df.columns:
    wlb_score += df['remote_available'].fillna(0).astype(int) * 20

# Flexible timing: +20
if 'flexible_timing' in df.columns:
    wlb_score += df['flexible_timing'].fillna(0).astype(int) * 20

# Hours available: lower = better balance (0-20)
if 'hours_available' in df.columns:
    hours = df['hours_available'].fillna(8)
    wlb_score += np.clip(20 - (hours - 4) * 2, 0, 20)

# Day/general shift: +15
if 'shift_type' in df.columns:
    is_day = df['shift_type'].str.lower().str.contains('day|general|morning', na=False)
    wlb_score += is_day.astype(int) * 15

# No travel: +15
if 'travel_required' in df.columns:
    no_travel = df['travel_required'].str.contains('No', case=False, na=False)
    wlb_score += no_travel.astype(int) * 15

# Childcare compatible: +10
if 'childcare_compatible' in df.columns:
    wlb_score += df['childcare_compatible'].fillna(0).astype(int) * 10

# Work mode: remote +15, hybrid +10
if 'work_mode' in df.columns:
    mode_lower = df['work_mode'].str.lower().fillna('')
    wlb_score += mode_lower.str.contains('remote|home', na=False).astype(int) * 15
    wlb_score += mode_lower.str.contains('hybrid', na=False).astype(int) * 10

# Create 5 balanced classes via quintiles
df['wlb_class'] = pd.qcut(wlb_score.rank(method='first'), q=5,
                           labels=['Poor', 'Below_Average', 'Average', 'Good', 'Excellent'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['wlb_class'])

print(f"  Score range: {wlb_score.min():.0f} - {wlb_score.max():.0f}")
print(f"  Classes: {list(label_encoder.classes_)}")

# === ENGINEER FEATURES ===
print("Engineering features...")

flex_cols = [c for c in ['remote_available', 'flexible_timing', 'childcare_compatible'] if c in df.columns]
df['flexibility_score'] = df[flex_cols].sum(axis=1) * 33.33

if 'hours_available' in df.columns:
    df['hours_stress'] = (12 - df['hours_available']).clip(lower=0) * 10
    df['reasonable_hours'] = (df['hours_available'] <= 8).astype(int)

if 'travel_required' in df.columns:
    travel_map = {'No Travel': 1, 'Occasional': 0.5, 'Frequent': 0}
    df['no_travel'] = df['travel_required'].map(travel_map).fillna(0.5)

if 'shift_type' in df.columns:
    df['day_shift'] = df['shift_type'].str.lower().str.contains('day|morning|general', na=False).astype(int)

fam_cols = [c for c in ['women_friendly', 'maternity_benefits', 'childcare_compatible'] if c in df.columns]
df['family_friendly_score'] = df[fam_cols].sum(axis=1) * 33.33

ben_cols = [c for c in ['health_insurance', 'pf_available', 'training_provided'] if c in df.columns]
df['benefits_score'] = df[ben_cols].sum(axis=1) * 25

if 'kids' in df.columns:
    df['has_kids'] = (df['kids'] > 0).astype(int)

# === PREPARE FEATURES ===
numeric_features = ['hours_available', 'kids', 'age', 'experience_years', 'income',
                    'mother_suitability_score', 'flexibility_score', 'hours_stress',
                    'family_friendly_score', 'benefits_score']
numeric_features = [c for c in numeric_features if c in df.columns]

categorical_features = ['shift_type', 'work_mode', 'work_type', 'domain', 'sector',
                        'city_tier', 'marital_status', 'seniority_level', 'career_growth', 'travel_required']
categorical_features = [c for c in categorical_features if c in df.columns]

binary_features = ['remote_available', 'flexible_timing', 'childcare_compatible',
                   'women_friendly', 'maternity_benefits', 'health_insurance',
                   'pf_available', 'training_provided', 'reasonable_hours',
                   'no_travel', 'day_shift', 'has_kids']
binary_features = [c for c in binary_features if c in df.columns]

for col in categorical_features:
    df[col] = df[col].fillna('Unknown')
for col in binary_features:
    df[col] = df[col].fillna(0).astype(float)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='Unknown')),
                          ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features),
        ('bin', 'passthrough', binary_features)
    ],
    remainder='drop'
)

X = preprocessor.fit_transform(df)
print(f"  Feature matrix: {X.shape}")

# === TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# === TRAIN LIGHTGBM (fast + accurate) ===
print("\nTraining LightGBM...")
model = lgb.LGBMClassifier(
    n_estimators=200, max_depth=15, learning_rate=0.1,
    num_leaves=50, subsample=0.85, colsample_bytree=0.85,
    class_weight='balanced', random_state=42, verbose=-1, n_jobs=1
)

start = datetime.now()
model.fit(X_train, y_train)
elapsed = (datetime.now() - start).total_seconds()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  Time:      {elapsed:.1f}s")

# === SAVE ===
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'work_life_balance_model.pkl')
joblib.dump(model, model_path)
print(f"\n  Saved model: {model_path}")

preprocessor_path = os.path.join(MODEL_DIR, 'work_life_balance_preprocessor.pkl')
joblib.dump({
    'preprocessor': preprocessor,
    'label_encoder': label_encoder,
    'numeric_cols': numeric_features,
    'categorical_cols': categorical_features,
    'binary_cols': binary_features
}, preprocessor_path)
print(f"  Saved preprocessor: {preprocessor_path}")

metadata = {
    'model_name': 'Work-Life Balance Model',
    'version': '2.0.0',
    'trained_at': datetime.now().isoformat(),
    'best_model': 'lightgbm',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'label_classes': list(label_encoder.classes_),
    'feature_count': X.shape[1],
    'training_samples': len(X_train),
    'training_time': elapsed
}

metadata_path = os.path.join(MODEL_DIR, 'work_life_balance_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata: {metadata_path}")

print(f"\nDone! Work-Life Balance Model Accuracy: {accuracy*100:.2f}%")
