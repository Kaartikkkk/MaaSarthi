#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MAASARTHI PROFILE COMPLETENESS SCORER                      ║
║              Evaluating & Scoring User Profile Quality                        ║
║                    Target Accuracy: 90%+                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

This model predicts profile quality level (High vs Needs-Improvement) based on
user profile inputs. The target is derived from career outcomes (skill_match,
career_growth, work_life_balance) to ensure meaningful quality assessment.

Author: MaaSarthi AI Team
Version: 3.0.0
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
class ProfileCompletenessConfig:
    """Configuration for Profile Completeness Scorer"""
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset.csv'))
    MODEL_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models'))
    
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_JOBS: int = -1
    
    # Target: derived profile quality (1=High, 0=NeedsImprovement)
    # Based on mother_suitability_score >= 8 (strong platform fit)
    TARGET_COL: str = 'profile_quality_target'
    
    # Columns used ONLY for deriving the target (excluded from features)
    TARGET_SOURCE_COLS: List[str] = field(default_factory=lambda: [
        'mother_suitability_score', 'skill_match_score',
        'career_growth', 'work_life_balance'
    ])
    
    # Profile fields and their weights for completeness scoring
    PROFILE_FIELDS: Dict[str, float] = field(default_factory=lambda: {
        'age': 10, 'education': 15, 'experience_years': 12,
        'primary_skill': 15, 'all_skills': 10, 'domain': 10,
        'city': 5, 'marital_status': 3, 'hours_available': 5,
        'work_mode': 5, 'secondary_skill': 3, 'language': 2,
        'device': 2, 'kids': 3
    })
    
    # Features the model can use (profile inputs only, no outcome scores)
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: [
        'age', 'experience_years', 'income', 'hours_available', 'kids',
        'salary_min', 'salary_max'
    ])
    
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        'education', 'domain', 'sector', 'seniority_level',
        'city_tier', 'work_mode', 'work_type', 'marital_status'
    ])
    
    BINARY_FEATURES: List[str] = field(default_factory=lambda: [
        'remote_available', 'flexible_timing', 'childcare_compatible',
        'women_friendly', 'maternity_benefits', 'training_provided',
        'health_insurance', 'pf_available'
    ])
    
    TEXT_FEATURES: List[str] = field(default_factory=lambda: [
        'all_skills', 'primary_skill', 'secondary_skill'
    ])


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
    classification_report, roc_auc_score, confusion_matrix
)


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE COMPLETENESS CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class ProfileCompletenessCalculator:
    """Rule-based profile completeness scorer"""
    
    def __init__(self, config: ProfileCompletenessConfig):
        self.config = config
        self.field_weights = config.PROFILE_FIELDS
        self.max_score = sum(self.field_weights.values())
    
    def calculate_score(self, profile: Dict) -> Dict:
        """Calculate completeness score for a profile"""
        score = 0
        filled_fields = []
        missing_fields = []
        field_scores = {}
        
        for field, weight in self.field_weights.items():
            value = profile.get(field)
            
            if self._is_filled(value):
                score += weight
                filled_fields.append(field)
                field_scores[field] = weight
            else:
                missing_fields.append(field)
                field_scores[field] = 0
        
        percentage = (score / self.max_score) * 100
        
        return {
            'raw_score': score,
            'max_score': self.max_score,
            'percentage': round(percentage, 1),
            'grade': self._get_grade(percentage),
            'filled_fields': filled_fields,
            'missing_fields': missing_fields,
            'field_scores': field_scores,
            'recommendations': self._get_recommendations(missing_fields, percentage)
        }
    
    def _is_filled(self, value) -> bool:
        """Check if a field value is filled"""
        if value is None:
            return False
        if isinstance(value, str) and value.strip() in ['', 'nan', 'None', 'null', 'Unknown']:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        return True
    
    def _get_grade(self, percentage: float) -> str:
        """Get grade based on percentage"""
        if percentage >= 90:
            return 'A+'
        elif percentage >= 80:
            return 'A'
        elif percentage >= 70:
            return 'B'
        elif percentage >= 60:
            return 'C'
        elif percentage >= 50:
            return 'D'
        else:
            return 'F'
    
    def _get_recommendations(self, missing_fields: List[str], percentage: float) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        # Priority order for missing fields
        priority_fields = ['primary_skill', 'education', 'experience_years', 'domain', 'all_skills']
        
        for field in priority_fields:
            if field in missing_fields:
                recommendations.append(f"Add your {field.replace('_', ' ')}")
        
        if percentage < 50:
            recommendations.insert(0, "Your profile is incomplete. Add more details to improve visibility.")
        elif percentage < 70:
            recommendations.insert(0, "Good start! A few more fields will boost your profile.")
        elif percentage < 90:
            recommendations.insert(0, "Almost complete! Add remaining fields for best matches.")
        
        return recommendations[:5]


# ══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class ProfileCompletenessDataProcessor:
    """Data processor for profile completeness prediction"""
    
    def __init__(self, config: ProfileCompletenessConfig):
        self.config = config
        self.preprocessor = None
        self.calculator = ProfileCompletenessCalculator(config)
        self.tfidf_skills = None
        self.svd = None
        
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
        
        # Ensure target source columns exist
        for col in self.config.TARGET_SOURCE_COLS:
            df = df.dropna(subset=[col])
        
        # Create derived target: profile quality = strong platform suitability
        # mother_suitability_score >= 8 indicates a highly suitable profile for MaaSarthi
        print("   Creating profile quality target...")
        df[self.config.TARGET_COL] = (df['mother_suitability_score'] >= 8).astype(int)
        
        print(f"   Target distribution: {Counter(df[self.config.TARGET_COL])}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer profile-based features"""
        print("🔧 Engineering profile quality features...")
        
        # Education ordinal encoding
        edu_order = {
            'Below 8th/Informal Education': 0, '8th Pass': 1,
            '10th Pass (SSC)': 2, '12th Pass (HSC)': 3,
            'Diploma/ITI': 4, 'Graduate (BTech/BA/BCom/BSc)': 5,
            'Post Graduate (MBA/MTech/MA/MSc)': 6, 'PhD/Doctorate': 7
        }
        df['education_level'] = df['education'].map(edu_order).fillna(3)
        
        # Seniority ordinal
        sen_order = {'Entry': 0, 'Junior': 1, 'Mid': 2, 'Senior': 3, 'Lead': 4, 'Manager': 5}
        df['seniority_num'] = df['seniority_level'].map(sen_order).fillna(1)
        
        # Skills count
        df['skills_count'] = df['all_skills'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() not in ['', 'nan'] else 0
        )
        
        # Has secondary skill
        df['has_secondary_skill'] = df['secondary_skill'].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip() in ['', 'nan', 'Unknown'] else 1
        )
        
        # Experience-education interaction
        df['exp_edu_interaction'] = df['experience_years'] * df['education_level']
        
        # Income per experience year (productivity proxy)
        df['income_per_exp'] = df['income'] / (df['experience_years'] + 1)
        
        # Benefits count
        benefit_cols = ['health_insurance', 'pf_available', 'maternity_benefits',
                       'training_provided', 'flexible_timing', 'childcare_compatible',
                       'women_friendly', 'remote_available']
        for col in benefit_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        df['benefits_count'] = df[[c for c in benefit_cols if c in df.columns]].sum(axis=1)
        
        # Hours available bucket
        df['hours_bucket'] = pd.cut(df['hours_available'], bins=[0, 3, 5, 7, 24],
                                     labels=[0, 1, 2, 3]).astype(float).fillna(1)
        
        # Age group
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 100],
                                  labels=[0, 1, 2, 3]).astype(float).fillna(1)
        
        # City tier numeric
        tier_map = {'Metro': 0, 'Tier-1': 1, 'Tier-2': 2, 'Tier-3': 3, 'Remote': 4, 'Rural': 5}
        df['city_tier_num'] = df['city_tier'].map(tier_map).fillna(2)
        
        # Drop target source columns from features
        for col in self.config.TARGET_SOURCE_COLS:
            if col in df.columns and col != self.config.TARGET_COL:
                df = df.drop(columns=[col], errors='ignore')
        
        # Also drop career_growth since it's a target source
        df = df.drop(columns=['career_growth'], errors='ignore')
        
        print(f"   Added engineered features")
        return df
    
    def create_preprocessor(self, df: pd.DataFrame):
        """Create preprocessing pipeline"""
        print("⚙️ Creating preprocessor...")
        
        numeric_cols = [c for c in self.config.NUMERIC_FEATURES if c in df.columns]
        categorical_cols = [c for c in self.config.CATEGORICAL_FEATURES if c in df.columns]
        binary_cols = [c for c in self.config.BINARY_FEATURES if c in df.columns]
        
        # Add engineered numeric features
        engineered_numeric = [
            'education_level', 'seniority_num', 'skills_count',
            'exp_edu_interaction', 'income_per_exp', 'benefits_count',
            'hours_bucket', 'age_group', 'city_tier_num'
        ]
        numeric_cols.extend([c for c in engineered_numeric if c in df.columns])
        
        # Add engineered binary features
        engineered_binary = ['has_secondary_skill']
        binary_cols.extend([c for c in engineered_binary if c in df.columns])
        
        # Fill missing for numeric
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill missing for categorical        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Fill missing for binary
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        print(f"   Numeric features: {len(numeric_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
        print(f"   Binary features: {len(binary_cols)}")
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
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
        """Process skills text"""
        print("📝 Processing skills text...")
        
        df['combined_skills'] = (
            df['all_skills'].fillna('') + ' ' +
            df['primary_skill'].fillna('') + ' ' +
            df['secondary_skill'].fillna('')
        )
        
        self.tfidf_skills = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=3,
            max_df=0.95
        )
        
        skills_tfidf = self.tfidf_skills.fit_transform(df['combined_skills'])
        
        self.svd = TruncatedSVD(n_components=15, random_state=self.config.RANDOM_STATE)
        text_reduced = self.svd.fit_transform(skills_tfidf)
        
        print(f"   Text features dimension: {text_reduced.shape[1]}")
        return text_reduced
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare final features"""
        
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
        y = df[self.config.TARGET_COL].values
        
        print(f"   Final feature matrix: {X.shape}")
        print(f"   Target distribution: {Counter(y)}")
        
        return X, y


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class ProfileCompletenessTrainer:
    """Trainer for profile completeness/verification prediction"""
    
    def __init__(self, config: ProfileCompletenessConfig):
        self.config = config
        self.models = {}
        self.ensemble_model = None
        self.best_model = None
        self.results = {}
        
    def build_models(self) -> Dict:
        """Build optimized models"""
        print("\n🏗️ Building optimized models for 90%+ accuracy...")
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=500,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE
            ),
            
            'xgboost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                eval_metric='logloss'
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=15,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                verbose=-1
            ),
            
            'hist_gradient_boosting': HistGradientBoostingClassifier(
                max_iter=500,
                max_depth=15,
                learning_rate=0.05,
                min_samples_leaf=10,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=self.config.RANDOM_STATE
            ),
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
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'training_time': elapsed
            }
            
            acc_color = "🟢" if accuracy >= 0.90 else "🟡" if accuracy >= 0.80 else "🔴"
            
            print(f"   {acc_color} Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   ROC-AUC: {roc_auc:.4f}")
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
            
            acc_color = "🟢" if scores.mean() >= 0.90 else "🟡"
            print(f"   {acc_color} Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def build_ensemble(self) -> StackingClassifier:
        """Build stacking ensemble"""
        print("\n🔗 Building Stacking Ensemble...")
        
        base_results = {k: v for k, v in self.results.items() if k in self.models}
        top_models = sorted(base_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:4]
        estimators = [(name, self.models[name]) for name, _ in top_models]
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
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
        
        print("🚀 Training Stacking Ensemble...")
        start_time = datetime.now()
        
        self.ensemble_model = self.build_ensemble()
        self.ensemble_model.fit(X_train, y_train)
        y_pred = self.ensemble_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.results['ensemble'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': elapsed
        }
        
        acc_color = "🟢" if accuracy >= 0.90 else "🟡"
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

class ProfileScorer:
    """Inference class for profile scoring"""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None,
                config: ProfileCompletenessConfig = None):
        self.config = config or ProfileCompletenessConfig()
        self.calculator = ProfileCompletenessCalculator(self.config)
        
        if model_path:
            self.model = joblib.load(model_path)
            self.preprocessor_data = joblib.load(preprocessor_path)
    
    def score_profile(self, profile: Dict) -> Dict:
        """Score a user profile"""
        # Rule-based completeness scoring
        completeness = self.calculator.calculate_score(profile)
        
        return {
            'completeness': completeness,
            'overall_grade': completeness['grade'],
            'percentage': completeness['percentage'],
            'improvements_needed': completeness['recommendations'],
            'ready_for_matching': completeness['percentage'] >= 70
        }
    
    def get_profile_summary(self, score_result: Dict) -> str:
        """Generate summary text"""
        pct = score_result['percentage']
        grade = score_result['overall_grade']
        
        if pct >= 90:
            return f"Excellent profile! (Grade: {grade}) Your profile is highly complete."
        elif pct >= 70:
            return f"Good profile! (Grade: {grade}) A few additions will improve your matches."
        elif pct >= 50:
            return f"Fair profile. (Grade: {grade}) Please complete more fields for better results."
        else:
            return f"Incomplete profile. (Grade: {grade}) Add essential information to get started."


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_profile_completeness_model(sample_frac: float = 1.0, skip_ensemble: bool = False):
    """Main training function"""
    
    config = ProfileCompletenessConfig()
    
    print("\n" + "="*70)
    print("📊 MAASARTHI PROFILE COMPLETENESS SCORER TRAINING")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Target Accuracy: 90%+")
    
    # Initialize
    processor = ProfileCompletenessDataProcessor(config)
    trainer = ProfileCompletenessTrainer(config)
    
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
    
    preprocessor_path = os.path.join(config.MODEL_DIR, 'profile_completeness_preprocessor.pkl')
    joblib.dump({
        'preprocessor': processor.preprocessor,
        'tfidf_skills': processor.tfidf_skills,
        'svd': processor.svd,
        'numeric_cols': processor.numeric_cols,
        'categorical_cols': processor.categorical_cols,
        'binary_cols': processor.binary_cols
    }, preprocessor_path)
    print(f"💾 Preprocessor saved to: {preprocessor_path}")
    
    model_path = os.path.join(config.MODEL_DIR, 'profile_completeness_model.pkl')
    joblib.dump(trainer.best_model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    best_name = max(trainer.results.items(), key=lambda x: x[1]['accuracy'])[0]
    metadata = {
        'model_name': 'Profile Completeness Scorer',
        'version': '3.0.0',
        'trained_at': datetime.now().isoformat(),
        'best_model': best_name,
        'accuracy': trainer.results[best_name]['accuracy'],
        'precision': trainer.results[best_name]['precision'],
        'recall': trainer.results[best_name]['recall'],
        'f1_score': trainer.results[best_name]['f1_score'],
        'roc_auc': trainer.results[best_name].get('roc_auc', 0),
        'feature_count': X.shape[1],
        'training_samples': len(X_train),
        'cv_results': cv_results,
        'all_results': trainer.results
    }
    
    metadata_path = os.path.join(config.MODEL_DIR, 'profile_completeness_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"📋 Metadata saved to: {metadata_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    final_acc = trainer.results[best_name]['accuracy']
    status = "🟢 TARGET MET!" if final_acc >= 0.90 else "🟡 Below target"
    
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
    
    parser = argparse.ArgumentParser(description='Train Profile Completeness Scorer')
    parser.add_argument('--sample', type=float, default=1.0,
                       help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble training')
    parser.add_argument('--demo', action='store_true',
                       help='Run scoring demo')
    
    args = parser.parse_args()
    
    if args.demo:
        scorer = ProfileScorer()
        sample_profile = {
            'age': 32,
            'education': 'Bachelor',
            'experience_years': 5,
            'primary_skill': 'Python',
            'all_skills': 'Python, SQL, Data Analysis',
            'domain': 'Technology',
            'city': 'Mumbai',
            'marital_status': 'Married'
        }
        result = scorer.score_profile(sample_profile)
        print(json.dumps(result, indent=2))
    else:
        print("📦 Checking dependencies...")
        results = train_profile_completeness_model(
            sample_frac=args.sample,
            skip_ensemble=args.skip_ensemble
        )
