#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MAASARTHI SKILL GAP ANALYZER MODEL                        ║
║               Identifying Skills Gaps & Training Recommendations             ║
║                    Target Accuracy: 85%+                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

This model analyzes the skill gap between a user's current skills and
job requirements, providing personalized training recommendations.

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
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SkillGapConfig:
    """Configuration for Skill Gap Analyzer"""
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset.csv'))
    MODEL_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models'))
    
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_JOBS: int = -1
    
    # Predict training_provided as proxy for skill gap
    # High skill gap -> training needed -> training_provided = 1
    TARGET_COL: str = 'training_provided'
    
    # Feature columns
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: [
        'age', 'experience_years', 'skill_match_score',
        'hours_available', 'mother_suitability_score', 'work_life_balance'
    ])
    
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        'domain', 'sector', 'seniority_level', 'education',
        'work_mode', 'city_tier', 'career_growth'
    ])
    
    TEXT_FEATURES: List[str] = field(default_factory=lambda: [
        'all_skills', 'primary_skill', 'secondary_skill', 'job_title'
    ])
    
    # TF-IDF settings
    TFIDF_MAX_FEATURES: int = 150
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

# Models - Binary Classification (training needed or not)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier,
    StackingClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix
)


# ══════════════════════════════════════════════════════════════════════════════
# SKILL KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

class SkillKnowledgeBase:
    """Knowledge base for skill relationships and recommendations"""
    
    def __init__(self):
        self.skill_domains = self._build_skill_domains()
        self.skill_prerequisites = self._build_prerequisites()
        self.skill_levels = self._build_skill_levels()
        
    def _build_skill_domains(self) -> Dict[str, List[str]]:
        """Map domains to required skills"""
        return {
            'Technology': ['python', 'java', 'sql', 'cloud', 'api', 'aws', 'docker', 'kubernetes'],
            'Data Science': ['python', 'machine learning', 'statistics', 'sql', 'tensorflow', 'pandas'],
            'Design': ['figma', 'photoshop', 'ui/ux', 'illustrator', 'sketch', 'prototyping'],
            'Marketing': ['seo', 'google ads', 'social media', 'content writing', 'analytics'],
            'Finance': ['excel', 'financial modeling', 'accounting', 'taxation', 'tally'],
            'Healthcare': ['patient care', 'medical records', 'hipaa', 'clinical research'],
            'Education': ['curriculum development', 'teaching', 'assessment', 'edtech'],
            'HR': ['recruitment', 'employee relations', 'hris', 'payroll', 'training'],
            'Sales': ['crm', 'negotiation', 'lead generation', 'closing', 'salesforce'],
            'Operations': ['process optimization', 'supply chain', 'quality control', 'lean']
        }
    
    def _build_prerequisites(self) -> Dict[str, List[str]]:
        """Map advanced skills to their prerequisites"""
        return {
            'machine learning': ['python', 'statistics', 'linear algebra'],
            'deep learning': ['machine learning', 'python', 'calculus'],
            'kubernetes': ['docker', 'linux', 'networking'],
            'microservices': ['api', 'docker', 'cloud'],
            'data engineering': ['sql', 'python', 'etl'],
            'cloud architecture': ['cloud', 'networking', 'security'],
            'devops': ['linux', 'git', 'ci/cd'],
            'full stack': ['frontend', 'backend', 'database']
        }
    
    def _build_skill_levels(self) -> Dict[str, Dict[str, List[str]]]:
        """Map skills to learning path levels"""
        return {
            'python': {
                'beginner': ['syntax', 'data types', 'functions'],
                'intermediate': ['oop', 'file handling', 'libraries'],
                'advanced': ['async', 'decorators', 'metaclasses']
            },
            'data science': {
                'beginner': ['statistics', 'excel', 'visualization'],
                'intermediate': ['python', 'pandas', 'scikit-learn'],
                'advanced': ['deep learning', 'mlops', 'deployment']
            }
        }
    
    def get_recommended_skills(self, domain: str, current_skills: List[str]) -> List[str]:
        """Get skill recommendations based on domain and current skills"""
        domain_skills = self.skill_domains.get(domain, [])
        current_lower = [s.lower() for s in current_skills]
        
        missing = [s for s in domain_skills if s.lower() not in current_lower]
        return missing[:5]  # Top 5 recommendations


# ══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class SkillGapDataProcessor:
    """Data processor for skill gap analysis"""
    
    def __init__(self, config: SkillGapConfig):
        self.config = config
        self.preprocessor = None
        self.tfidf_skills = None
        self.tfidf_jobs = None
        self.svd = None
        self.skill_kb = SkillKnowledgeBase()
        
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
        
        # Ensure target exists
        df = df.dropna(subset=[self.config.TARGET_COL])
        
        # Fill missing
        for col in self.config.NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        for col in self.config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        for col in self.config.TEXT_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer skill-gap specific features"""
        print("🔧 Engineering skill gap features...")
        
        # Skills count
        df['skills_count'] = df['all_skills'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0
        )
        
        # Skill diversity (unique skills / total)
        df['skill_diversity'] = df['all_skills'].apply(
            lambda x: len(set(str(x).lower().split(','))) / max(len(str(x).split(',')), 1)
            if pd.notna(x) else 0
        )
        
        # Experience-skill ratio (skills per year of experience)
        df['skill_exp_ratio'] = df['skills_count'] / (df['experience_years'] + 1)
        
        # Skill match gap (inverse of skill_match_score)
        if 'skill_match_score' in df.columns:
            df['skill_gap_indicator'] = 100 - df['skill_match_score']
        
        # Career growth potential gap (convert categorical to numeric first)
        if 'career_growth' in df.columns:
            growth_map = {'Low': 30, 'Medium': 60, 'High': 90}
            df['career_growth_num'] = df['career_growth'].map(growth_map).fillna(50)
            df['growth_gap'] = 100 - df['career_growth_num']
        
        # Seniority-skill alignment
        seniority_skill_map = {
            'Entry': 3, 'Junior': 6, 'Mid': 10, 'Senior': 15, 'Lead': 20, 'Manager': 15
        }
        if 'seniority_level' in df.columns:
            df['expected_skills'] = df['seniority_level'].map(seniority_skill_map).fillna(5)
            df['skill_deficiency'] = (df['expected_skills'] - df['skills_count']).clip(lower=0)
        
        # Domain complexity (some domains require more skills)
        domain_complexity = {
            'Technology': 1.3, 'Data Science': 1.4, 'Finance': 1.2,
            'Healthcare': 1.3, 'Marketing': 1.1, 'Education': 1.0
        }
        if 'domain' in df.columns:
            df['domain_complexity'] = df['domain'].map(domain_complexity).fillna(1.0)
        
        # Training likelihood score
        df['training_likelihood'] = (
            df.get('skill_deficiency', 0) * 5 +
            df.get('skill_gap_indicator', 50) * 0.5 +
            df.get('growth_gap', 50) * 0.3
        )
        
        print(f"   Added 9 engineered features")
        return df
    
    def create_preprocessor(self, df: pd.DataFrame):
        """Create preprocessing pipeline"""
        print("⚙️ Creating preprocessor...")
        
        numeric_cols = [c for c in self.config.NUMERIC_FEATURES if c in df.columns]
        categorical_cols = [c for c in self.config.CATEGORICAL_FEATURES if c in df.columns]
        
        # Add engineered features
        engineered = ['skills_count', 'skill_diversity', 'skill_exp_ratio',
                     'skill_gap_indicator', 'growth_gap', 'expected_skills',
                     'skill_deficiency', 'domain_complexity', 'training_likelihood']
        numeric_cols.extend([c for c in engineered if c in df.columns])
        
        print(f"   Numeric features: {len(numeric_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
        
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
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop',
            n_jobs=self.config.N_JOBS
        )
        
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
    
    def process_text_features(self, df: pd.DataFrame) -> np.ndarray:
        """Process skills text with TF-IDF"""
        print("📝 Processing skills text features...")
        
        # Combine skills
        df['combined_skills'] = (
            df['all_skills'].fillna('') + ' ' +
            df['primary_skill'].fillna('') + ' ' +
            df['secondary_skill'].fillna('')
        )
        
        # TF-IDF for skills
        self.tfidf_skills = TfidfVectorizer(
            max_features=self.config.TFIDF_MAX_FEATURES,
            ngram_range=self.config.TFIDF_NGRAM_RANGE,
            stop_words='english',
            min_df=3,
            max_df=0.95
        )
        
        skills_tfidf = self.tfidf_skills.fit_transform(df['combined_skills'])
        
        # TF-IDF for job titles
        self.tfidf_jobs = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=3,
            max_df=0.95
        )
        
        jobs_tfidf = self.tfidf_jobs.fit_transform(df['job_title'].fillna(''))
        
        # Combine and reduce
        text_features = hstack([skills_tfidf, jobs_tfidf])
        
        self.svd = TruncatedSVD(n_components=40, random_state=self.config.RANDOM_STATE)
        text_reduced = self.svd.fit_transform(text_features)
        
        print(f"   Text features dimension: {text_reduced.shape[1]}")
        return text_reduced
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare final features and target"""
        
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
        y = df[self.config.TARGET_COL].astype(int).values
        
        print(f"   Final feature matrix: {X.shape}")
        print(f"   Target distribution: {Counter(y)}")
        
        return X, y


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class SkillGapTrainer:
    """Trainer for skill gap analyzer (binary classification)"""
    
    def __init__(self, config: SkillGapConfig):
        self.config = config
        self.models = {}
        self.ensemble_model = None
        self.best_model = None
        self.results = {}
        
    def build_models(self) -> Dict:
        """Build optimized models for binary classification"""
        print("\n🏗️ Building optimized models for 85%+ accuracy...")
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=400,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight='balanced',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=400,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight='balanced',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE
            ),
            
            'xgboost': xgb.XGBClassifier(
                n_estimators=400,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=1,
                scale_pos_weight=1,
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=15,
                learning_rate=0.1,
                num_leaves=50,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight='balanced',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                verbose=-1
            ),
            
            'hist_gradient_boosting': HistGradientBoostingClassifier(
                max_iter=400,
                max_depth=12,
                learning_rate=0.1,
                min_samples_leaf=10,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.config.RANDOM_STATE
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.85,
                min_samples_split=3,
                random_state=self.config.RANDOM_STATE
            ),
            
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.config.RANDOM_STATE
            ),
            
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                class_weight='balanced',
                n_jobs=self.config.N_JOBS,
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
            
            acc_color = "🟢" if accuracy >= 0.85 else "🟡" if accuracy >= 0.75 else "🔴"
            
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

class SkillGapAnalyzer:
    """Inference class for skill gap analysis"""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        if model_path:
            self.model = joblib.load(model_path)
            self.preprocessor_data = joblib.load(preprocessor_path)
        self.skill_kb = SkillKnowledgeBase()
    
    def analyze_skills(self, user_skills: List[str], target_domain: str,
                      target_role: str = None) -> Dict:
        """Analyze skill gaps for a user"""
        
        # Get domain required skills
        domain_skills = self.skill_kb.skill_domains.get(target_domain, [])
        
        # Find missing skills
        user_skills_lower = [s.lower().strip() for s in user_skills]
        missing_skills = [s for s in domain_skills if s.lower() not in user_skills_lower]
        
        # Get prerequisites for advanced skills
        prerequisites_needed = []
        for skill in missing_skills:
            prereqs = self.skill_kb.skill_prerequisites.get(skill, [])
            for prereq in prereqs:
                if prereq.lower() not in user_skills_lower:
                    prerequisites_needed.append(prereq)
        
        # Calculate skill gap score
        total_required = len(domain_skills)
        skills_matched = total_required - len(missing_skills)
        gap_score = (len(missing_skills) / max(total_required, 1)) * 100
        
        return {
            'domain': target_domain,
            'current_skills': user_skills,
            'skills_count': len(user_skills),
            'required_skills': domain_skills,
            'missing_skills': missing_skills[:5],  # Top 5
            'prerequisites_needed': list(set(prerequisites_needed))[:3],
            'skill_match_percentage': (skills_matched / max(total_required, 1)) * 100,
            'gap_score': gap_score,
            'training_recommended': gap_score > 30,
            'priority_skills': missing_skills[:3] if missing_skills else [],
            'recommendations': self._get_recommendations(gap_score)
        }
    
    def _get_recommendations(self, gap_score: float) -> List[str]:
        """Get recommendations based on gap score"""
        if gap_score < 20:
            return [
                "Your skills are well-aligned with domain requirements",
                "Consider advanced certifications to stand out",
                "Focus on soft skills development"
            ]
        elif gap_score < 40:
            return [
                "Minor skill gaps detected",
                "Consider online courses in missing areas",
                "Practice projects can help bridge the gap"
            ]
        elif gap_score < 60:
            return [
                "Moderate skill gaps detected",
                "Structured learning recommended",
                "Consider bootcamps or certification programs",
                "Target 2-3 key skills first"
            ]
        else:
            return [
                "Significant skill gaps detected",
                "Comprehensive training recommended",
                "Start with foundational skills",
                "Consider career coaching",
                "Set 6-month learning goals"
            ]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_skill_gap_model(sample_frac: float = 1.0, skip_ensemble: bool = False):
    """Main training function"""
    
    config = SkillGapConfig()
    
    print("\n" + "="*70)
    print("📚 MAASARTHI SKILL GAP ANALYZER MODEL TRAINING")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Target Accuracy: 85%+")
    
    # Initialize
    processor = SkillGapDataProcessor(config)
    trainer = SkillGapTrainer(config)
    
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
    
    preprocessor_path = os.path.join(config.MODEL_DIR, 'skill_gap_preprocessor.pkl')
    joblib.dump({
        'preprocessor': processor.preprocessor,
        'tfidf_skills': processor.tfidf_skills,
        'tfidf_jobs': processor.tfidf_jobs,
        'svd': processor.svd,
        'numeric_cols': processor.numeric_cols,
        'categorical_cols': processor.categorical_cols
    }, preprocessor_path)
    print(f"💾 Preprocessor saved to: {preprocessor_path}")
    
    model_path = os.path.join(config.MODEL_DIR, 'skill_gap_model.pkl')
    joblib.dump(trainer.best_model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    best_name = max(trainer.results.items(), key=lambda x: x[1]['accuracy'])[0]
    metadata = {
        'model_name': 'Skill Gap Analyzer',
        'version': '2.0.0',
        'trained_at': datetime.now().isoformat(),
        'best_model': best_name,
        'accuracy': trainer.results[best_name]['accuracy'],
        'precision': trainer.results[best_name]['precision'],
        'recall': trainer.results[best_name]['recall'],
        'f1_score': trainer.results[best_name]['f1_score'],
        'feature_count': X.shape[1],
        'training_samples': len(X_train),
        'cv_results': cv_results,
        'all_results': trainer.results
    }
    
    metadata_path = os.path.join(config.MODEL_DIR, 'skill_gap_metadata.json')
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
    
    parser = argparse.ArgumentParser(description='Train Skill Gap Analyzer')
    parser.add_argument('--sample', type=float, default=1.0,
                       help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble training')
    parser.add_argument('--analyze', action='store_true',
                       help='Run skill gap analysis demo')
    
    args = parser.parse_args()
    
    if args.analyze:
        # Demo analysis
        analyzer = SkillGapAnalyzer()
        result = analyzer.analyze_skills(
            user_skills=['python', 'sql', 'excel'],
            target_domain='Data Science'
        )
        print(json.dumps(result, indent=2))
    else:
        print("📦 Checking dependencies...")
        results = train_skill_gap_model(
            sample_frac=args.sample,
            skip_ensemble=args.skip_ensemble
        )
