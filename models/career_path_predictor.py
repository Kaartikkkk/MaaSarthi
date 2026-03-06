#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MAASARTHI CAREER PATH PREDICTOR                           ║
║             Predicting Career Trajectory & Seniority Progression             ║
║                    Target Accuracy: 85%+                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

This model predicts the career path and seniority progression for users
based on their profile, skills, experience, and domain.

Author: MaaSarthi AI Team
Version: 2.0.0
"""

import os
import sys
import json
import warnings
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CareerPathConfig:
    """Configuration for Career Path Predictor"""
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset.csv'))
    MODEL_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models'))
    
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_JOBS: int = -1
    
    # Target: seniority_level (multi-class classification)
    TARGET_COL: str = 'seniority_level'
    
    # Feature columns
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: [
        'age', 'experience_years', 'income', 'skill_match_score',
        'work_life_balance', 'hours_available', 'kids'
    ])
    
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        'domain', 'sector', 'education', 'work_mode', 'work_type',
        'city_tier', 'marital_status', 'career_growth'
    ])
    
    BINARY_FEATURES: List[str] = field(default_factory=lambda: [
        'remote_available', 'flexible_timing', 'training_provided',
        'health_insurance', 'pf_available', 'is_verified'
    ])
    
    TEXT_FEATURES: List[str] = field(default_factory=lambda: [
        'all_skills', 'primary_skill', 'job_title'
    ])
    
    TFIDF_MAX_FEATURES: int = 100
    TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

# Models
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier, StackingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


# ══════════════════════════════════════════════════════════════════════════════
# CAREER PATH KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

class CareerPathKnowledgeBase:
    """Knowledge base for career progression"""
    
    def __init__(self):
        self.seniority_order = ['Entry', 'Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director', 'VP', 'CXO']
        self.typical_progression = self._build_progression()
        self.domain_paths = self._build_domain_paths()
        
    def _build_progression(self) -> Dict[str, Dict]:
        """Build typical career progression patterns"""
        return {
            'Entry': {'next': 'Junior', 'years_needed': 1, 'skills_needed': 3},
            'Junior': {'next': 'Mid', 'years_needed': 2, 'skills_needed': 5},
            'Mid': {'next': 'Senior', 'years_needed': 3, 'skills_needed': 8},
            'Senior': {'next': 'Lead', 'years_needed': 3, 'skills_needed': 10},
            'Lead': {'next': 'Manager', 'years_needed': 2, 'skills_needed': 12},
            'Manager': {'next': 'Director', 'years_needed': 4, 'skills_needed': 15},
            'Director': {'next': 'VP', 'years_needed': 5, 'skills_needed': 18},
            'VP': {'next': 'CXO', 'years_needed': 5, 'skills_needed': 20}
        }
    
    def _build_domain_paths(self) -> Dict[str, List[str]]:
        """Build domain-specific career paths"""
        return {
            'Technology': [
                'Developer -> Senior Developer -> Tech Lead -> Architect -> CTO',
                'Developer -> DevOps -> Platform Engineer -> Infrastructure Director'
            ],
            'Data Science': [
                'Data Analyst -> Data Scientist -> Senior DS -> Lead DS -> Head of Data',
                'ML Engineer -> Senior ML Engineer -> ML Architect -> Chief AI Officer'
            ],
            'Design': [
                'UI Designer -> Senior Designer -> Design Lead -> Design Director -> CDO',
                'UX Researcher -> Senior UX -> UX Manager -> VP of Design'
            ],
            'Marketing': [
                'Marketing Executive -> Manager -> Director -> CMO',
                'Content Writer -> Content Lead -> Content Director -> VP Marketing'
            ],
            'Finance': [
                'Analyst -> Senior Analyst -> Manager -> Director -> CFO',
                'Accountant -> Senior Accountant -> Controller -> Finance Director'
            ]
        }
    
    def predict_next_role(self, current_level: str, experience_years: int,
                         skills_count: int) -> Dict:
        """Predict next career step"""
        if current_level not in self.typical_progression:
            return {'next_role': 'Unknown', 'readiness': 0}
        
        prog = self.typical_progression[current_level]
        
        # Calculate readiness
        years_ready = min(experience_years / prog['years_needed'], 1.0)
        skills_ready = min(skills_count / prog['skills_needed'], 1.0)
        readiness = (years_ready * 0.5 + skills_ready * 0.5) * 100
        
        return {
            'current_level': current_level,
            'next_role': prog['next'],
            'readiness_percentage': round(readiness, 1),
            'years_to_next': max(0, prog['years_needed'] - experience_years),
            'skills_gap': max(0, prog['skills_needed'] - skills_count)
        }


# ══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class CareerPathDataProcessor:
    """Data processor for career path prediction"""
    
    def __init__(self, config: CareerPathConfig):
        self.config = config
        self.preprocessor = None
        self.label_encoder = None
        self.tfidf_skills = None
        self.tfidf_jobs = None
        self.svd = None
        self.career_kb = CareerPathKnowledgeBase()
        
    def load_data(self, sample_frac: float = 1.0) -> pd.DataFrame:
        """Load dataset"""
        print(f"📂 Loading data from: {self.config.DATA_PATH}")
        df = pd.read_csv(self.config.DATA_PATH)
        print(f"   Total records: {len(df):,}")
        
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=self.config.RANDOM_STATE)
            print(f"   Sampled records: {len(df):,}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data"""
        print("🧹 Cleaning data...")
        
        initial_len = len(df)
        df = df.drop_duplicates()
        print(f"   Removed {initial_len - len(df)} duplicates")
        
        # Remove missing targets
        df = df.dropna(subset=[self.config.TARGET_COL])
        
        # Standardize seniority levels
        seniority_map = {
            'entry': 'Entry', 'entry level': 'Entry', 'fresher': 'Entry', 'intern': 'Entry',
            'junior': 'Junior', 'associate': 'Junior',
            'mid': 'Mid', 'mid-level': 'Mid', 'intermediate': 'Mid',
            'senior': 'Senior', 'sr': 'Senior',
            'lead': 'Lead', 'team lead': 'Lead', 'principal': 'Lead',
            'manager': 'Manager', 'mgr': 'Manager',
            'director': 'Director', 'head': 'Director',
            'vp': 'VP', 'vice president': 'VP',
            'c-level': 'CXO', 'cxo': 'CXO', 'executive': 'CXO'
        }
        
        df['seniority_clean'] = df[self.config.TARGET_COL].str.lower().map(
            lambda x: seniority_map.get(x, x.title() if pd.notna(x) else 'Unknown')
        )
        
        # Fill missing values
        for col in self.config.NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        for col in self.config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        for col in self.config.BINARY_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        for col in self.config.TEXT_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        return df
    
    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare target variable"""
        print("📊 Preparing target classes...")
        
        # Use the cleaned seniority
        self.label_encoder = LabelEncoder()
        df['target_encoded'] = self.label_encoder.fit_transform(df['seniority_clean'])
        
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        print(f"   Distribution:")
        for cls in self.label_encoder.classes_:
            count = (df['seniority_clean'] == cls).sum()
            pct = count / len(df) * 100
            print(f"      {cls}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer career-related features"""
        print("🔧 Engineering career features...")
        
        # Skills count
        df['skills_count'] = df['all_skills'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0
        )
        
        # Experience-age ratio (career start age indicator)
        df['exp_age_ratio'] = df['experience_years'] / (df['age'] - 17).clip(lower=1)
        
        # Career velocity (skills per year)
        df['career_velocity'] = df['skills_count'] / (df['experience_years'] + 1)
        
        # Income per experience year
        if 'income' in df.columns:
            df['income_per_exp'] = df['income'] / (df['experience_years'] + 1)
        
        # Education level encoding
        edu_levels = {
            '10th': 1, '12th': 2, 'diploma': 3, 'graduate': 4, 'bachelor': 4,
            'postgraduate': 5, 'master': 5, 'mba': 5, 'phd': 6, 'doctorate': 6
        }
        if 'education' in df.columns:
            df['education_level'] = df['education'].str.lower().map(
                lambda x: max([edu_levels.get(k, 3) for k in str(x).split() if k in edu_levels], default=3)
            )
        
        # Career growth potential (convert categorical to numeric first)
        if 'career_growth' in df.columns and 'experience_years' in df.columns:
            growth_map = {'Low': 30, 'Medium': 60, 'High': 90}
            df['career_growth_num'] = df['career_growth'].map(growth_map).fillna(50)
            df['growth_potential'] = df['career_growth_num'] / (df['experience_years'] + 1)
        
        # Work stability indicator
        stability_cols = ['is_verified', 'health_insurance', 'pf_available']
        available = [c for c in stability_cols if c in df.columns]
        if available:
            df['job_stability'] = df[available].sum(axis=1)
        
        # Leadership indicator (from work mode and type)
        if 'work_mode' in df.columns:
            df['leadership_indicator'] = df['work_mode'].str.contains(
                'lead|manage|director', case=False, na=False
            ).astype(int)
        
        print(f"   Added 8 engineered features")
        return df
    
    def create_preprocessor(self, df: pd.DataFrame):
        """Create preprocessing pipeline"""
        print("⚙️ Creating preprocessor...")
        
        numeric_cols = [c for c in self.config.NUMERIC_FEATURES if c in df.columns]
        categorical_cols = [c for c in self.config.CATEGORICAL_FEATURES if c in df.columns]
        binary_cols = [c for c in self.config.BINARY_FEATURES if c in df.columns]
        
        # Add engineered features
        engineered_numeric = ['skills_count', 'exp_age_ratio', 'career_velocity',
                             'income_per_exp', 'education_level', 'growth_potential',
                             'job_stability', 'leadership_indicator']
        numeric_cols.extend([c for c in engineered_numeric if c in df.columns])
        
        print(f"   Numeric features: {len(numeric_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
        print(f"   Binary features: {len(binary_cols)}")
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols),
                ('bin', 'passthrough', binary_cols)
            ],
            remainder='drop',
            n_jobs=self.config.N_JOBS
        )
        
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols
    
    def process_text_features(self, df: pd.DataFrame) -> np.ndarray:
        """Process text features"""
        print("📝 Processing text features...")
        
        # Combine skills
        df['combined_text'] = (
            df['all_skills'].fillna('') + ' ' +
            df['primary_skill'].fillna('') + ' ' +
            df['job_title'].fillna('')
        )
        
        self.tfidf_skills = TfidfVectorizer(
            max_features=self.config.TFIDF_MAX_FEATURES,
            ngram_range=self.config.TFIDF_NGRAM_RANGE,
            stop_words='english',
            min_df=3,
            max_df=0.95
        )
        
        text_tfidf = self.tfidf_skills.fit_transform(df['combined_text'])
        
        # Dimensionality reduction
        self.svd = TruncatedSVD(n_components=25, random_state=self.config.RANDOM_STATE)
        text_reduced = self.svd.fit_transform(text_tfidf)
        
        print(f"   Text features dimension: {text_reduced.shape[1]}")
        return text_reduced
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare final features"""
        
        # Prepare target
        df = self.prepare_target(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Create preprocessor
        self.create_preprocessor(df)
        
        # Transform structured features
        X_structured = self.preprocessor.fit_transform(df)
        
        # Process text
        X_text = self.process_text_features(df)
        
        # Combine
        if hasattr(X_structured, 'toarray'):
            X_structured = X_structured.toarray()
        
        X = np.hstack([X_structured, X_text])
        y = df['target_encoded'].values
        
        print(f"   Final feature matrix: {X.shape}")
        print(f"   Target classes: {len(np.unique(y))}")
        
        return X, y


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class CareerPathTrainer:
    """Trainer for career path prediction"""
    
    def __init__(self, config: CareerPathConfig):
        self.config = config
        self.models = {}
        self.ensemble_model = None
        self.best_model = None
        self.results = {}
        
    def build_models(self) -> Dict:
        """Build optimized models"""
        print("\n🏗️ Building optimized models for 85%+ accuracy...")
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=400,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight='balanced',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=400,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight='balanced',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE
            ),
            
            'xgboost': xgb.XGBClassifier(
                n_estimators=400,
                max_depth=12,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=1,
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=18,
                learning_rate=0.08,
                num_leaves=60,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight='balanced',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                verbose=-1
            ),
            
            'hist_gradient_boosting': HistGradientBoostingClassifier(
                max_iter=400,
                max_depth=15,
                learning_rate=0.08,
                min_samples_leaf=10,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.config.RANDOM_STATE
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.08,
                subsample=0.85,
                min_samples_split=3,
                random_state=self.config.RANDOM_STATE
            ),
            
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0005,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=600,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=25,
                random_state=self.config.RANDOM_STATE
            )
        }
        
        for name in models:
            print(f"   ✓ {name.replace('_', ' ').title()} configured")
        
        return models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train all models"""
        
        print("\n" + "="*70)
        print("TRAINING INDIVIDUAL MODELS")
        print("="*70)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")
            start_time = datetime.now()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': elapsed
            }
            
            acc_color = "🟢" if accuracy >= 0.85 else "🟡" if accuracy >= 0.75 else "🔴"
            
            print(f"   {acc_color} Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   Training time: {elapsed:.1f}s")
        
        self.results = results
        return results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Cross-validation"""
        print("\n" + "="*70)
        print(f"CROSS-VALIDATION ({self.config.CV_FOLDS}-FOLD)")
        print("="*70)
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True,
                             random_state=self.config.RANDOM_STATE)
        
        base_results = {k: v for k, v in self.results.items() if k in self.models}
        top_models = sorted(base_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
        
        for name, _ in top_models:
            model = self.models[name]
            print(f"\n📊 Cross-validating {name}...")
            
            scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy',
                                    n_jobs=self.config.N_JOBS)
            
            cv_results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores.tolist()
            }
            
            acc_color = "🟢" if scores.mean() >= 0.85 else "🟡"
            print(f"   {acc_color} Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def build_ensemble(self) -> VotingClassifier:
        """Build voting ensemble"""
        print("\n🔗 Building Voting Ensemble...")
        
        base_results = {k: v for k, v in self.results.items() if k in self.models}
        top_models = sorted(base_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:5]
        estimators = [(name, self.models[name]) for name, _ in top_models]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=self.config.N_JOBS
        )
        
        print(f"   Ensemble with {len(estimators)} models")
        return ensemble
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray):
        """Train ensemble"""
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE")
        print("="*70)
        
        print("🚀 Training Voting Ensemble...")
        start_time = datetime.now()
        
        self.ensemble_model = self.build_ensemble()
        self.ensemble_model.fit(X_train, y_train)
        y_pred = self.ensemble_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.results['ensemble'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': elapsed
        }
        
        acc_color = "🟢" if accuracy >= 0.85 else "🟡"
        print(f"\n   {acc_color} Ensemble Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Training time: {elapsed:.1f}s")
        
        # Select best 
        best_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        if best_name == 'ensemble':
            self.best_model = self.ensemble_model
        else:
            self.best_model = self.models[best_name]
        
        print(f"\n   🏆 Best Model: {best_name} ({self.results[best_name]['accuracy']:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

class CareerPathPredictor:
    """Inference class for career path prediction"""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None,
                metadata_path: str = None):
        if model_path:
            self.model = joblib.load(model_path)
            self.preprocessor_data = joblib.load(preprocessor_path)
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        self.career_kb = CareerPathKnowledgeBase()
    
    def predict_career_path(self, user_profile: Dict) -> Dict:
        """Predict career path for a user"""
        current_level = user_profile.get('seniority_level', 'Entry')
        experience = user_profile.get('experience_years', 0)
        skills_count = len(user_profile.get('skills', []))
        domain = user_profile.get('domain', 'Technology')
        
        # Get next role prediction
        next_role = self.career_kb.predict_next_role(current_level, experience, skills_count)
        
        # Get domain-specific paths
        domain_paths = self.career_kb.domain_paths.get(domain, [])
        
        return {
            'current_level': current_level,
            'predicted_next_level': next_role['next_role'],
            'readiness_percentage': next_role['readiness_percentage'],
            'years_to_promotion': next_role['years_to_next'],
            'skills_gap': next_role['skills_gap'],
            'domain': domain,
            'suggested_paths': domain_paths[:2] if domain_paths else [],
            'recommendations': self._get_recommendations(next_role)
        }
    
    def _get_recommendations(self, next_role: Dict) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if next_role['readiness_percentage'] >= 80:
            recommendations.append("You're ready for the next level!")
            recommendations.append("Start applying for senior positions")
            recommendations.append("Consider mentoring others")
        elif next_role['readiness_percentage'] >= 50:
            recommendations.append(f"Focus on gaining {next_role['skills_gap']} more key skills")
            recommendations.append("Take on leadership responsibilities")
            recommendations.append("Build your network")
        else:
            recommendations.append("Focus on skill development")
            recommendations.append("Seek mentorship from seniors")
            recommendations.append(f"Gain {next_role['years_to_next']} more years of experience")
        
        return recommendations


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_career_path_model(sample_frac: float = 1.0, skip_ensemble: bool = False):
    """Main training function"""
    
    config = CareerPathConfig()
    
    print("\n" + "="*70)
    print("🚀 MAASARTHI CAREER PATH PREDICTOR TRAINING")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Target Accuracy: 85%+")
    
    # Initialize
    processor = CareerPathDataProcessor(config)
    trainer = CareerPathTrainer(config)
    
    # Step 1: Load and process
    print("\n" + "-"*50)
    print("STEP 1: DATA PROCESSING")
    print("-"*50)
    
    df = processor.load_data(sample_frac)
    df = processor.clean_data(df)
    X, y = processor.prepare_data(df)
    
    # Step 2: Split
    print("\n" + "-"*50)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("-"*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        stratify=y, random_state=config.RANDOM_STATE
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # Step 3: Train
    print("\n" + "-"*50)
    print("STEP 3: MODEL TRAINING")
    print("-"*50)
    
    trainer.models = trainer.build_models()
    trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Step 4: CV
    print("\n" + "-"*50)
    print("STEP 4: CROSS-VALIDATION")
    print("-"*50)
    
    cv_results = trainer.cross_validate(X, y)
    
    # Step 5: Ensemble
    if not skip_ensemble:
        print("\n" + "-"*50)
        print("STEP 5: ENSEMBLE TRAINING")
        print("-"*50)
        
        trainer.train_ensemble(X_train, y_train, X_test, y_test)
    
    # Step 6: Save
    print("\n" + "-"*50)
    print("STEP 6: SAVING MODELS")
    print("-"*50)
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    preprocessor_path = os.path.join(config.MODEL_DIR, 'career_path_preprocessor.pkl')
    joblib.dump({
        'preprocessor': processor.preprocessor,
        'label_encoder': processor.label_encoder,
        'tfidf_skills': processor.tfidf_skills,
        'svd': processor.svd,
        'numeric_cols': processor.numeric_cols,
        'categorical_cols': processor.categorical_cols,
        'binary_cols': processor.binary_cols
    }, preprocessor_path)
    print(f"💾 Preprocessor saved to: {preprocessor_path}")
    
    model_path = os.path.join(config.MODEL_DIR, 'career_path_model.pkl')
    joblib.dump(trainer.best_model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    best_name = max(trainer.results.items(), key=lambda x: x[1]['accuracy'])[0]
    metadata = {
        'model_name': 'Career Path Predictor',
        'version': '2.0.0',
        'trained_at': datetime.now().isoformat(),
        'best_model': best_name,
        'accuracy': trainer.results[best_name]['accuracy'],
        'precision': trainer.results[best_name]['precision'],
        'recall': trainer.results[best_name]['recall'],
        'f1_score': trainer.results[best_name]['f1_score'],
        'label_classes': list(processor.label_encoder.classes_),
        'feature_count': X.shape[1],
        'training_samples': len(X_train),
        'cv_results': cv_results,
        'all_results': trainer.results
    }
    
    metadata_path = os.path.join(config.MODEL_DIR, 'career_path_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"📋 Metadata saved to: {metadata_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    final_acc = trainer.results[best_name]['accuracy']
    status = "🟢 TARGET MET!" if final_acc >= 0.85 else "🟡 Below target"
    
    print(f"\n   Best Model: {best_name}")
    print(f"   Final Accuracy: {final_acc:.4f} {status}")
    print(f"   Precision: {trainer.results[best_name]['precision']:.4f}")
    print(f"   Recall: {trainer.results[best_name]['recall']:.4f}")
    print(f"   F1-Score: {trainer.results[best_name]['f1_score']:.4f}")
    print(f"\n   Model saved to: {model_path}")
    print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return trainer.results


# ══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Career Path Predictor')
    parser.add_argument('--sample', type=float, default=1.0,
                       help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble training')
    
    args = parser.parse_args()
    
    print("📦 Checking dependencies...")
    results = train_career_path_model(
        sample_frac=args.sample,
        skip_ensemble=args.skip_ensemble
    )
