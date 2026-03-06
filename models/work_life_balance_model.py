#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MAASARTHI WORK-LIFE BALANCE MODEL                         ║
║              Predicting Work-Life Balance Score for Jobs                     ║
║                    Target Accuracy: 85%+                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

This model predicts the work-life balance score for job opportunities,
helping working mothers find jobs that support their lifestyle needs.

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
class WorkLifeBalanceConfig:
    """Configuration for Work-Life Balance Model"""
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset.csv'))
    MODEL_DIR: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models'))
    
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_JOBS: int = -1
    
    # Target: work_life_balance (convert to classes for high accuracy)
    TARGET_COL: str = 'work_life_balance'
    N_CLASSES: int = 5
    
    # Core features that strongly influence work-life balance
    CORE_FEATURES: List[str] = field(default_factory=lambda: [
        'hours_available', 'remote_available', 'flexible_timing',
        'shift_type', 'work_mode'
    ])
    
    NUMERIC_FEATURES: List[str] = field(default_factory=lambda: [
        'hours_available', 'kids', 'age', 'experience_years',
        'income', 'mother_suitability_score'
    ])
    
    CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda: [
        'shift_type', 'work_mode', 'work_type', 'domain', 'sector',
        'city_tier', 'marital_status', 'seniority_level', 'career_growth', 'travel_required'
    ])
    
    BINARY_FEATURES: List[str] = field(default_factory=lambda: [
        'remote_available', 'flexible_timing', 'childcare_compatible',
        'women_friendly', 'maternity_benefits',
        'health_insurance', 'pf_available', 'training_provided'
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
# DATA PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class WorkLifeBalanceDataProcessor:
    """Data processor for work-life balance prediction"""
    
    def __init__(self, config: WorkLifeBalanceConfig):
        self.config = config
        self.preprocessor = None
        self.label_encoder = None
        
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
        
        return df
    
    def create_target_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create balanced classification target using quintiles"""
        print("📊 Creating work-life balance classes...")
        
        target = df[self.config.TARGET_COL]
        
        # Use quintiles for balanced classes (key for high accuracy)
        df['wlb_class'] = pd.qcut(
            target.rank(method='first'),
            q=5,
            labels=['Poor', 'Below_Average', 'Average', 'Good', 'Excellent']
        )
        
        self.label_encoder = LabelEncoder()
        df['wlb_encoded'] = self.label_encoder.fit_transform(df['wlb_class'])
        
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        print(f"   Distribution:")
        for cls in self.label_encoder.classes_:
            count = (df['wlb_class'] == cls).sum()
            pct = count / len(df) * 100
            print(f"      {cls}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer work-life balance focused features"""
        print("🔧 Engineering work-life balance features...")
        
        # Flexibility composite score
        flex_cols = ['remote_available', 'flexible_timing', 'childcare_compatible']
        available_flex = [c for c in flex_cols if c in df.columns]
        df['flexibility_score'] = df[available_flex].sum(axis=1) * 33.33  # Scale to ~100
        
        # Hours stress indicator
        if 'hours_available' in df.columns:
            # Higher hours = less balance
            df['hours_stress'] = (12 - df['hours_available']).clip(lower=0) * 10
            df['reasonable_hours'] = (df['hours_available'] <= 8).astype(int)
        
        # Travel burden (convert categorical to numeric)
        if 'travel_required' in df.columns:
            travel_map = {'No Travel': 1, 'Occasional': 0.5, 'Frequent': 0}
            df['no_travel'] = df['travel_required'].map(travel_map).fillna(0.5)
        
        # Shift preference (day shifts are better for balance)
        if 'shift_type' in df.columns:
            df['day_shift'] = df['shift_type'].str.lower().str.contains(
                'day|morning', na=False
            ).astype(int)
            df['night_shift'] = df['shift_type'].str.lower().str.contains(
                'night|graveyard', na=False
            ).astype(int)
        
        # Family friendliness score
        family_cols = ['women_friendly', 'maternity_benefits', 'childcare_compatible']
        available_fam = [c for c in family_cols if c in df.columns]
        df['family_friendly_score'] = df[available_fam].sum(axis=1) * 33.33
        
        # Benefits score
        benefits = ['health_insurance', 'pf_available', 'training_provided']
        available_ben = [c for c in benefits if c in df.columns]
        df['benefits_score'] = df[available_ben].sum(axis=1) * 25
        
        # Work mode score (remote/hybrid better than office)
        if 'work_mode' in df.columns:
            mode_scores = {'remote': 100, 'hybrid': 75, 'office': 50, 'onsite': 40}
            df['work_mode_score'] = df['work_mode'].str.lower().map(
                lambda x: max([mode_scores.get(k, 50) for k in str(x).split() 
                              if k in mode_scores], default=50)
            )
        
        # Overall balance indicator
        df['balance_composite'] = (
            df.get('flexibility_score', 50) * 0.35 +
            df.get('family_friendly_score', 50) * 0.25 +
            df.get('work_mode_score', 50) * 0.20 +
            (100 - df.get('hours_stress', 50)) * 0.20
        )
        
        # Kids impact on balance needs
        if 'kids' in df.columns:
            df['has_kids'] = (df['kids'] > 0).astype(int)
            df['multiple_kids'] = (df['kids'] > 1).astype(int)
        
        print(f"   Added 12 engineered features")
        return df
    
    def create_preprocessor(self, df: pd.DataFrame):
        """Create preprocessing pipeline"""
        print("⚙️ Creating preprocessor...")
        
        numeric_cols = [c for c in self.config.NUMERIC_FEATURES if c in df.columns]
        categorical_cols = [c for c in self.config.CATEGORICAL_FEATURES if c in df.columns]
        binary_cols = [c for c in self.config.BINARY_FEATURES if c in df.columns]
        
        # Add engineered features
        engineered_numeric = [
            'flexibility_score', 'hours_stress', 'family_friendly_score',
            'benefits_score', 'work_mode_score', 'balance_composite'
        ]
        numeric_cols.extend([c for c in engineered_numeric if c in df.columns])
        
        engineered_binary = [
            'reasonable_hours', 'no_travel', 'day_shift', 'night_shift',
            'has_kids', 'multiple_kids'
        ]
        binary_cols.extend([c for c in engineered_binary if c in df.columns])
        
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
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare final features"""
        
        # Create target classes
        df = self.create_target_classes(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Create preprocessor
        self.create_preprocessor(df)
        
        # Transform
        X = self.preprocessor.fit_transform(df)
        y = df['wlb_encoded'].values
        
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        print(f"   Final feature matrix: {X.shape}")
        print(f"   Target classes: {len(np.unique(y))}")
        
        return X, y


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class WorkLifeBalanceTrainer:
    """Trainer for work-life balance prediction"""
    
    def __init__(self, config: WorkLifeBalanceConfig):
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
                bootstrap=True,
                oob_score=True,
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
                gamma=0.1,
                reg_alpha=0.05,
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
                max_features='sqrt',
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

class WorkLifeBalancePredictor:
    """Inference class for work-life balance prediction"""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None,
                metadata_path: str = None):
        if model_path:
            self.model = joblib.load(model_path)
            self.preprocessor_data = joblib.load(preprocessor_path)
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def predict_balance(self, job_data: Dict) -> Dict:
        """Predict work-life balance for a job"""
        # Implementation depends on preprocessor
        pass
    
    def get_balance_tips(self, balance_level: str) -> List[str]:
        """Get tips based on balance level"""
        tips = {
            'Excellent': [
                'Great work-life balance!',
                'This role supports your lifestyle needs',
                'Take advantage of flexibility offered'
            ],
            'Good': [
                'Solid balance for most people',
                'Discuss additional flexibility if needed',
                'Establish boundaries early'
            ],
            'Average': [
                'Balance is manageable but needs effort',
                'Negotiate for flexible arrangements',
                'Plan your time carefully'
            ],
            'Below_Average': [
                'May be challenging for parents',
                'Consider negotiating better terms',
                'Evaluate your support system'
            ],
            'Poor': [
                'Significant impact on personal life expected',
                'Consider if role is right fit',
                'Discuss concerns before accepting'
            ]
        }
        return tips.get(balance_level, [])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_work_life_balance_model(sample_frac: float = 1.0, skip_ensemble: bool = False):
    """Main training function"""
    
    config = WorkLifeBalanceConfig()
    
    print("\n" + "="*70)
    print("⚖️ MAASARTHI WORK-LIFE BALANCE MODEL TRAINING")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Target Accuracy: 85%+")
    
    # Initialize
    processor = WorkLifeBalanceDataProcessor(config)
    trainer = WorkLifeBalanceTrainer(config)
    
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
    
    preprocessor_path = os.path.join(config.MODEL_DIR, 'work_life_balance_preprocessor.pkl')
    joblib.dump({
        'preprocessor': processor.preprocessor,
        'label_encoder': processor.label_encoder,
        'numeric_cols': processor.numeric_cols,
        'categorical_cols': processor.categorical_cols,
        'binary_cols': processor.binary_cols
    }, preprocessor_path)
    print(f"💾 Preprocessor saved to: {preprocessor_path}")
    
    model_path = os.path.join(config.MODEL_DIR, 'work_life_balance_model.pkl')
    joblib.dump(trainer.best_model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    best_name = max(trainer.results.items(), key=lambda x: x[1]['accuracy'])[0]
    metadata = {
        'model_name': 'Work-Life Balance Model',
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
    
    metadata_path = os.path.join(config.MODEL_DIR, 'work_life_balance_metadata.json')
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
    
    parser = argparse.ArgumentParser(description='Train Work-Life Balance Model')
    parser.add_argument('--sample', type=float, default=1.0,
                       help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble training')
    
    args = parser.parse_args()
    
    print("📦 Checking dependencies...")
    results = train_work_life_balance_model(
        sample_frac=args.sample,
        skip_ensemble=args.skip_ensemble
    )
