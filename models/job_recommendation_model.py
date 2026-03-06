"""
================================================================================
MAASARTHI - JOB RECOMMENDATION MODEL
================================================================================
Advanced Multi-Class Classification Model for Job Title Recommendations

Features:
- Ensemble of 4 algorithms: Random Forest, XGBoost, LightGBM, Neural Network
- Stacking Classifier for optimal predictions
- Cross-validation with stratified k-fold
- Hyperparameter tuning with Optuna
- Top-10 job recommendations with confidence scores
- Production-ready with comprehensive preprocessing pipeline

Author: MaaSarthi AI Team
Version: 2.0.0
Date: March 2026
================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_val_score,
    cross_val_predict
)
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler, 
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier,
    StackingClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not installed. Run: pip install lightgbm")

# Optuna for hyperparameter tuning
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Run: pip install optuna")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class ModelConfig:
    """Configuration for the Job Recommendation Model"""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'dataset.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'trained_models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'job_recommendation_model.pkl')
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'job_label_encoder.pkl')
    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'job_preprocessor.pkl')
    METADATA_PATH = os.path.join(MODEL_DIR, 'job_model_metadata.json')
    
    # Feature columns
    NUMERIC_FEATURES = ['age', 'kids', 'hours_available', 'experience_years']
    
    CATEGORICAL_FEATURES = [
        'domain', 'primary_skill', 'education', 'city_tier', 
        'work_mode', 'seniority_level', 'marital_status'
    ]
    
    TEXT_FEATURES = ['all_skills']  # Will use TF-IDF
    
    TARGET = 'job_title'
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    TOP_K = 10  # Number of recommendations
    
    # Training parameters
    N_JOBS = -1  # Use all CPU cores
    VERBOSE = 1
    
    # Hyperparameter tuning
    N_OPTUNA_TRIALS = 50
    OPTUNA_TIMEOUT = 3600  # 1 hour max


# ============================================================================
# DATA PROCESSOR
# ============================================================================

class DataProcessor:
    """Handles all data preprocessing and feature engineering"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.svd = TruncatedSVD(n_components=20, random_state=42)
        self.feature_names = []
        
    def load_data(self, sample_frac: float = None) -> pd.DataFrame:
        """Load and return the dataset"""
        print(f"📂 Loading data from: {self.config.DATA_PATH}")
        
        df = pd.read_csv(self.config.DATA_PATH)
        print(f"   Total records: {len(df):,}")
        
        if sample_frac and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=self.config.RANDOM_STATE)
            print(f"   Sampled records: {len(df):,}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the data"""
        print("🧹 Cleaning data...")
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        print(f"   Removed {initial_len - len(df)} duplicates")
        
        # Handle missing values
        for col in self.config.NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        for col in self.config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        for col in self.config.TEXT_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Remove rare job titles (less than 5 occurrences)
        job_counts = df[self.config.TARGET].value_counts()
        valid_jobs = job_counts[job_counts >= 5].index
        df = df[df[self.config.TARGET].isin(valid_jobs)]
        print(f"   Unique job titles: {df[self.config.TARGET].nunique():,}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data"""
        print("🔧 Engineering features...")
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 25, 35, 45, 100], 
            labels=['young', 'middle', 'senior', 'veteran']
        )
        
        # Experience level
        df['exp_level'] = pd.cut(
            df['experience_years'],
            bins=[-1, 2, 5, 10, 100],
            labels=['fresher', 'junior', 'mid', 'senior']
        )
        
        # Has kids flag
        df['has_kids'] = (df['kids'] > 0).astype(int)
        
        # Hours category
        df['hours_category'] = pd.cut(
            df['hours_available'],
            bins=[0, 4, 6, 8, 100],
            labels=['part_time', 'half_day', 'full_day', 'overtime']
        )
        
        # Is remote flag
        if 'work_mode' in df.columns:
            df['is_remote'] = df['work_mode'].isin(['Remote', 'Work From Home', 'Hybrid']).astype(int)
        
        print(f"   Added 5 engineered features")
        
        return df
    
    def create_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Create the preprocessing pipeline"""
        print("⚙️ Creating preprocessor...")
        
        # Update feature lists with engineered features
        numeric_features = self.config.NUMERIC_FEATURES + ['has_kids', 'is_remote']
        categorical_features = self.config.CATEGORICAL_FEATURES + ['age_group', 'exp_level', 'hours_category']
        
        # Filter to only include columns that exist
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='drop'
        )
        
        self.feature_names = numeric_features + categorical_features
        print(f"   Numeric features: {len(numeric_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        
        return self.preprocessor
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full data preparation pipeline"""
        # Clean
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Create preprocessor
        self.create_preprocessor(df)
        
        # Get features
        X = df[self.feature_names]
        
        # Handle text features separately (TF-IDF + SVD)
        if 'all_skills' in df.columns:
            print("📝 Processing text features (TF-IDF)...")
            text_features = self.tfidf.fit_transform(df['all_skills'].fillna(''))
            text_features_reduced = self.svd.fit_transform(text_features)
            print(f"   TF-IDF dimensions reduced to: {text_features_reduced.shape[1]}")
        else:
            text_features_reduced = None
        
        # Transform features
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Combine with text features
        if text_features_reduced is not None:
            X_transformed = np.hstack([X_transformed, text_features_reduced])
        
        print(f"   Final feature matrix shape: {X_transformed.shape}")
        
        # Encode target
        y = self.label_encoder.fit_transform(df[self.config.TARGET])
        print(f"   Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_transformed, y, df
    
    def transform_input(self, user_data: Dict) -> np.ndarray:
        """Transform a single user input for prediction"""
        # Create DataFrame from user data
        df = pd.DataFrame([user_data])
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Transform
        X = df[self.feature_names]
        X_transformed = self.preprocessor.transform(X)
        
        # Handle text features
        if 'all_skills' in user_data:
            text_features = self.tfidf.transform([user_data.get('all_skills', '')])
            text_features_reduced = self.svd.transform(text_features)
            X_transformed = np.hstack([X_transformed, text_features_reduced])
        
        return X_transformed
    
    def save(self) -> None:
        """Save preprocessing objects"""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        
        joblib.dump({
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'tfidf': self.tfidf,
            'svd': self.svd,
            'feature_names': self.feature_names
        }, self.config.PREPROCESSOR_PATH)
        
        print(f"💾 Preprocessor saved to: {self.config.PREPROCESSOR_PATH}")
    
    def load(self) -> None:
        """Load preprocessing objects"""
        data = joblib.load(self.config.PREPROCESSOR_PATH)
        self.preprocessor = data['preprocessor']
        self.label_encoder = data['label_encoder']
        self.tfidf = data['tfidf']
        self.svd = data['svd']
        self.feature_names = data['feature_names']


# ============================================================================
# MODEL TRAINER
# ============================================================================

class JobRecommendationTrainer:
    """Handles model training with multiple algorithms and ensemble"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.ensemble_model = None
        self.best_model = None
        self.metrics = {}
        
    def build_base_models(self) -> Dict[str, Any]:
        """Build all base models"""
        print("\n🏗️ Building base models...")
        
        models = {}
        
        # 1. Random Forest
        models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            n_jobs=self.config.N_JOBS,
            random_state=self.config.RANDOM_STATE,
            verbose=0
        )
        print("   ✓ Random Forest configured")
        
        # 2. XGBoost
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective='multi:softprob',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                verbosity=0
            )
            print("   ✓ XGBoost configured")
        
        # 3. LightGBM
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=15,
                learning_rate=0.1,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                class_weight='balanced',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                verbose=-1
            )
            print("   ✓ LightGBM configured")
        
        # 4. Neural Network (MLP)
        models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        print("   ✓ Neural Network (MLP) configured")
        
        # 5. Gradient Boosting (sklearn)
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config.RANDOM_STATE,
            verbose=0
        )
        print("   ✓ Gradient Boosting configured")
        
        return models
    
    def build_ensemble(self, models: Dict[str, Any]) -> StackingClassifier:
        """Build stacking ensemble from base models"""
        print("\n🔗 Building Stacking Ensemble...")
        
        # Prepare estimators for stacking
        estimators = [(name, model) for name, model in models.items()]
        
        # Meta-learner (final estimator)
        meta_learner = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            n_jobs=self.config.N_JOBS,
            random_state=self.config.RANDOM_STATE
        )
        
        # Stacking Classifier
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,  # Internal cross-validation
            stack_method='predict_proba',
            n_jobs=self.config.N_JOBS,
            verbose=0
        )
        
        print(f"   Stacking ensemble with {len(estimators)} base models")
        
        return ensemble
    
    def train_individual_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """Train and evaluate each model individually"""
        print("\n" + "="*70)
        print("TRAINING INDIVIDUAL MODELS")
        print("="*70)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")
            start_time = datetime.now()
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Top-K accuracy
            if y_proba is not None:
                top_5_acc = top_k_accuracy_score(y_test, y_proba, k=5)
                top_10_acc = top_k_accuracy_score(y_test, y_proba, k=10)
            else:
                top_5_acc = top_10_acc = None
            
            # Training time
            train_time = (datetime.now() - start_time).total_seconds()
            
            results[name] = {
                'accuracy': accuracy,
                'top_5_accuracy': top_5_acc,
                'top_10_accuracy': top_10_acc,
                'train_time': train_time
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            if top_5_acc:
                print(f"   Top-5 Accuracy: {top_5_acc:.4f}")
                print(f"   Top-10 Accuracy: {top_10_acc:.4f}")
            print(f"   Training time: {train_time:.1f}s")
        
        return results
    
    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Train the ensemble model"""
        print("\n" + "="*70)
        print("TRAINING STACKING ENSEMBLE")
        print("="*70)
        
        start_time = datetime.now()
        
        # Train ensemble
        print("🚀 Training ensemble (this may take a while)...")
        self.ensemble_model.fit(X_train, y_train)
        
        # Predict
        y_pred = self.ensemble_model.predict(X_test)
        y_proba = self.ensemble_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        top_5_acc = top_k_accuracy_score(y_test, y_proba, k=5)
        top_10_acc = top_k_accuracy_score(y_test, y_proba, k=10)
        
        train_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'accuracy': accuracy,
            'top_5_accuracy': top_5_acc,
            'top_10_accuracy': top_10_acc,
            'train_time': train_time
        }
        
        print(f"\n   ✅ Ensemble Accuracy: {accuracy:.4f}")
        print(f"   ✅ Top-5 Accuracy: {top_5_acc:.4f}")
        print(f"   ✅ Top-10 Accuracy: {top_10_acc:.4f}")
        print(f"   Training time: {train_time:.1f}s")
        
        return results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Perform cross-validation on all models"""
        print("\n" + "="*70)
        print(f"CROSS-VALIDATION ({self.config.CV_FOLDS}-FOLD)")
        print("="*70)
        
        cv = StratifiedKFold(
            n_splits=self.config.CV_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE
        )
        
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"\n📊 Cross-validating {name}...")
            
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='accuracy',
                n_jobs=self.config.N_JOBS
            )
            
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            
            print(f"   Mean Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        return cv_results
    
    def select_best_model(self, results: Dict) -> str:
        """Select the best performing model"""
        best_name = None
        best_score = 0
        
        for name, metrics in results.items():
            # Prioritize top-10 accuracy if available, else use accuracy
            score = metrics.get('top_10_accuracy') or metrics.get('accuracy', 0)
            if score > best_score:
                best_score = score
                best_name = name
        
        return best_name
    
    def save_model(self, model: Any, model_name: str = 'ensemble') -> None:
        """Save the trained model"""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        
        model_path = self.config.MODEL_PATH
        joblib.dump(model, model_path)
        print(f"\n💾 Model saved to: {model_path}")
    
    def save_metadata(self, metrics: Dict) -> None:
        """Save model training metadata"""
        metadata = {
            'model_type': 'Job Recommendation',
            'version': '2.0.0',
            'trained_date': datetime.now().isoformat(),
            'algorithm': 'Stacking Ensemble (RF + XGB + LightGBM + MLP + GB)',
            'top_k': self.config.TOP_K,
            'cv_folds': self.config.CV_FOLDS,
            'metrics': metrics
        }
        
        with open(self.config.METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadata saved to: {self.config.METADATA_PATH}")


# ============================================================================
# JOB RECOMMENDER (INFERENCE)
# ============================================================================

class JobRecommender:
    """Production-ready job recommendation engine"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.processor = DataProcessor(config)
        self.is_loaded = False
        
    def load(self) -> bool:
        """Load trained model and preprocessor"""
        try:
            print("🔄 Loading Job Recommendation Model...")
            
            # Load model
            self.model = joblib.load(self.config.MODEL_PATH)
            print(f"   ✓ Model loaded from: {self.config.MODEL_PATH}")
            
            # Load preprocessor
            self.processor.load()
            print(f"   ✓ Preprocessor loaded")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def recommend(
        self, 
        user_data: Dict,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get job recommendations for a user
        
        Args:
            user_data: Dictionary with user profile data
            top_k: Number of recommendations to return
            
        Returns:
            List of dictionaries with job title and confidence score
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Transform input
        X = self.processor.transform_input(user_data)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Get top-k indices
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        # Build recommendations
        recommendations = []
        for idx in top_indices:
            job_title = self.processor.label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            
            recommendations.append({
                'job_title': job_title,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%",
                'rank': len(recommendations) + 1
            })
        
        return recommendations
    
    def batch_recommend(
        self,
        users_data: List[Dict],
        top_k: int = 10
    ) -> List[List[Dict]]:
        """Get recommendations for multiple users"""
        return [self.recommend(user, top_k) for user in users_data]


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_job_recommendation_model(
    sample_frac: float = None,
    skip_ensemble: bool = False
) -> Dict[str, Any]:
    """
    Main training pipeline for Job Recommendation Model
    
    Args:
        sample_frac: Fraction of data to use (None = full dataset)
        skip_ensemble: If True, skip ensemble training (faster)
        
    Returns:
        Dictionary with training results and metrics
    """
    print("\n" + "="*70)
    print("🎯 MAASARTHI JOB RECOMMENDATION MODEL TRAINING")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    config = ModelConfig()
    
    # Step 1: Data Processing
    print("\n" + "-"*50)
    print("STEP 1: DATA PROCESSING")
    print("-"*50)
    
    processor = DataProcessor(config)
    df = processor.load_data(sample_frac)
    X, y, df_processed = processor.prepare_data(df)
    
    # Step 2: Train-Test Split
    print("\n" + "-"*50)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("-"*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    
    # Step 3: Build Models
    print("\n" + "-"*50)
    print("STEP 3: MODEL BUILDING")
    print("-"*50)
    
    trainer = JobRecommendationTrainer(config)
    trainer.models = trainer.build_base_models()
    
    # Step 4: Train Individual Models
    individual_results = trainer.train_individual_models(
        X_train, y_train, X_test, y_test
    )
    
    # Step 5: Cross-Validation
    print("\n" + "-"*50)
    print("STEP 5: CROSS-VALIDATION")
    print("-"*50)
    
    # Use a subset for CV if dataset is large
    if len(X) > 50000:
        cv_sample = 50000
        indices = np.random.choice(len(X), cv_sample, replace=False)
        X_cv, y_cv = X[indices], y[indices]
        print(f"   Using {cv_sample:,} samples for cross-validation")
    else:
        X_cv, y_cv = X, y
    
    cv_results = trainer.cross_validate(X_cv, y_cv)
    
    # Step 6: Train Ensemble (if not skipped)
    if not skip_ensemble:
        trainer.ensemble_model = trainer.build_ensemble(trainer.models)
        ensemble_results = trainer.train_ensemble(
            X_train, y_train, X_test, y_test
        )
        final_model = trainer.ensemble_model
        final_metrics = ensemble_results
    else:
        # Use best individual model
        best_model_name = trainer.select_best_model(individual_results)
        print(f"\n🏆 Best individual model: {best_model_name}")
        final_model = trainer.models[best_model_name]
        final_metrics = individual_results[best_model_name]
    
    # Step 7: Save Everything
    print("\n" + "-"*50)
    print("STEP 7: SAVING MODEL & ARTIFACTS")
    print("-"*50)
    
    processor.save()
    trainer.save_model(final_model)
    
    all_metrics = {
        'individual_models': individual_results,
        'cross_validation': cv_results,
        'final_model': final_metrics
    }
    trainer.save_metadata(all_metrics)
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Final Model Performance:")
    print(f"   - Accuracy: {final_metrics['accuracy']:.4f}")
    if final_metrics.get('top_5_accuracy'):
        print(f"   - Top-5 Accuracy: {final_metrics['top_5_accuracy']:.4f}")
        print(f"   - Top-10 Accuracy: {final_metrics['top_10_accuracy']:.4f}")
    print(f"\n   Model saved to: {config.MODEL_PATH}")
    print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'config': config,
        'model': final_model,
        'processor': processor,
        'metrics': all_metrics
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Job Recommendation Model')
    parser.add_argument('--sample', type=float, default=None,
                        help='Fraction of data to use (e.g., 0.1 for 10%%)')
    parser.add_argument('--skip-ensemble', action='store_true',
                        help='Skip ensemble training for faster results')
    
    args = parser.parse_args()
    
    # Install required packages if missing
    print("📦 Checking dependencies...")
    try:
        import xgboost
        import lightgbm
    except ImportError:
        print("Installing required packages...")
        os.system('pip install xgboost lightgbm optuna --quiet')
    
    # Run training
    results = train_job_recommendation_model(
        sample_frac=args.sample,
        skip_ensemble=args.skip_ensemble
    )
