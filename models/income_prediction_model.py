"""
================================================================================
MAASARTHI - INCOME PREDICTION MODEL
================================================================================
Advanced Regression Model for Salary/Income Prediction

Features:
- Ensemble of 5 algorithms: Random Forest, XGBoost, LightGBM, Neural Network, 
  Gradient Boosting
- Stacking Regressor for optimal predictions
- Cross-validation with K-Fold
- Comprehensive evaluation metrics (MAE, RMSE, R², MAPE)
- Salary bracket classification
- Confidence intervals for predictions
- Production-ready with preprocessing pipeline

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
from typing import List, Dict, Tuple, Optional, Any, Union

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, 
    KFold, 
    cross_val_score,
    cross_val_predict
)
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler, 
    OneHotEncoder,
    RobustScaler,
    QuantileTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, 
    VotingRegressor,
    StackingRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import (
    Ridge, 
    ElasticNet,
    HuberRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score
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

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class IncomeModelConfig:
    """Configuration for the Income Prediction Model"""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'dataset.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'trained_models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'income_prediction_model.pkl')
    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'income_preprocessor.pkl')
    METADATA_PATH = os.path.join(MODEL_DIR, 'income_model_metadata.json')
    
    # Feature columns
    NUMERIC_FEATURES = [
        'age', 'kids', 'hours_available', 'experience_years',
        'mother_suitability_score', 'skill_match_score', 'work_life_balance'
    ]
    
    CATEGORICAL_FEATURES = [
        'domain', 'sector', 'primary_skill', 'education', 'city', 'city_tier',
        'state', 'work_mode', 'work_type', 'seniority_level', 'marital_status',
        'shift_type', 'career_growth'
    ]
    
    BINARY_FEATURES = [
        'remote_available', 'flexible_timing', 'childcare_compatible',
        'women_friendly', 'maternity_benefits', 'training_provided',
        'health_insurance', 'pf_available'
    ]
    
    TEXT_FEATURES = ['all_skills', 'job_title']
    
    TARGET = 'income'
    
    # Salary brackets for classification
    SALARY_BRACKETS = [
        (0, 10000, 'Below 10K'),
        (10000, 20000, '10K-20K'),
        (20000, 35000, '20K-35K'),
        (35000, 50000, '35K-50K'),
        (50000, 75000, '50K-75K'),
        (75000, 100000, '75K-1L'),
        (100000, 150000, '1L-1.5L'),
        (150000, 200000, '1.5L-2L'),
        (200000, float('inf'), 'Above 2L')
    ]
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Training parameters
    N_JOBS = -1
    VERBOSE = 1
    
    # Outlier handling
    INCOME_MIN = 5000     # Minimum valid income
    INCOME_MAX = 500000   # Maximum valid income


# ============================================================================
# DATA PROCESSOR
# ============================================================================

class IncomeDataProcessor:
    """Handles all data preprocessing for income prediction"""
    
    def __init__(self, config: IncomeModelConfig = None):
        self.config = config or IncomeModelConfig()
        self.preprocessor = None
        self.tfidf_skills = TfidfVectorizer(max_features=50, stop_words='english')
        self.tfidf_jobs = TfidfVectorizer(max_features=30, stop_words='english')
        self.svd_skills = TruncatedSVD(n_components=15, random_state=42)
        self.svd_jobs = TruncatedSVD(n_components=10, random_state=42)
        self.target_scaler = RobustScaler()  # For income normalization
        self.feature_names = []
        self.income_stats = {}
        
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
        
        initial_len = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"   Removed {initial_len - len(df)} duplicates")
        
        # Handle missing values in numeric columns
        for col in self.config.NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Handle missing values in categorical columns
        for col in self.config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Handle binary features
        for col in self.config.BINARY_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(int)
        
        # Handle text features
        for col in self.config.TEXT_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Remove income outliers
        if self.config.TARGET in df.columns:
            before_outlier = len(df)
            df = df[
                (df[self.config.TARGET] >= self.config.INCOME_MIN) & 
                (df[self.config.TARGET] <= self.config.INCOME_MAX)
            ]
            print(f"   Removed {before_outlier - len(df)} income outliers")
            
            # Store income statistics
            self.income_stats = {
                'min': float(df[self.config.TARGET].min()),
                'max': float(df[self.config.TARGET].max()),
                'mean': float(df[self.config.TARGET].mean()),
                'median': float(df[self.config.TARGET].median()),
                'std': float(df[self.config.TARGET].std())
            }
            print(f"   Income range: ₹{self.income_stats['min']:,.0f} - ₹{self.income_stats['max']:,.0f}")
            print(f"   Mean income: ₹{self.income_stats['mean']:,.0f}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data"""
        print("🔧 Engineering features...")
        
        feature_count = 0
        
        # 1. Age-related features
        df['age_squared'] = df['age'] ** 2  # Capture non-linear age effect
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 30, 35, 40, 50, 100],
            labels=['<25', '25-30', '30-35', '35-40', '40-50', '50+']
        )
        feature_count += 2
        
        # 2. Experience-related features
        df['exp_squared'] = df['experience_years'] ** 2
        df['exp_age_ratio'] = df['experience_years'] / (df['age'] - 18 + 1)  # Experience efficiency
        df['exp_level'] = pd.cut(
            df['experience_years'],
            bins=[-1, 1, 3, 5, 8, 12, 100],
            labels=['fresher', 'junior', 'mid', 'senior', 'lead', 'expert']
        )
        feature_count += 3
        
        # 3. Work-life features
        df['flexibility_score'] = (
            df['remote_available'].astype(int) +
            df['flexible_timing'].astype(int) +
            (df['work_mode'].isin(['Remote', 'Work From Home', 'Hybrid'])).astype(int)
        )
        feature_count += 1
        
        # 4. Benefits score
        benefits_cols = ['health_insurance', 'pf_available', 'maternity_benefits', 'training_provided']
        existing_benefits = [col for col in benefits_cols if col in df.columns]
        if existing_benefits:
            df['benefits_score'] = df[existing_benefits].sum(axis=1)
            feature_count += 1
        
        # 5. Mother-friendliness score (composite)
        mother_cols = ['mother_suitability_score', 'childcare_compatible', 'women_friendly', 'flexible_timing']
        existing_mother = [col for col in mother_cols if col in df.columns]
        if len(existing_mother) >= 2:
            df['mother_friendly_composite'] = df[existing_mother].mean(axis=1)
            feature_count += 1
        
        # 6. Location tier encoding
        tier_map = {'Metro': 4, 'Tier-1': 3, 'Tier-2': 2, 'Tier-3': 1, 'Rural': 0, 'Remote': 3}
        if 'city_tier' in df.columns:
            df['city_tier_num'] = df['city_tier'].map(tier_map).fillna(1)
            feature_count += 1
        
        # 7. Has kids impact
        df['has_kids'] = (df['kids'] > 0).astype(int)
        df['kids_impact'] = df['kids'] * df['hours_available']  # Kids reducing available hours effect
        feature_count += 2
        
        # 8. Seniority numeric encoding
        seniority_map = {
            'Entry Level/Fresher': 0, 'Associate': 1, 'Senior Associate': 2,
            'Manager': 3, 'Senior Manager': 4, 'Director/Executive': 5
        }
        if 'seniority_level' in df.columns:
            df['seniority_num'] = df['seniority_level'].map(seniority_map).fillna(1)
            feature_count += 1
        
        # 9. Career growth encoding
        growth_map = {'Low': 0, 'Medium': 1, 'High': 2}
        if 'career_growth' in df.columns:
            df['career_growth_num'] = df['career_growth'].map(growth_map).fillna(1)
            feature_count += 1
        
        # 10. Education level encoding
        edu_map = {
            'Below 8th/Informal Education': 0, '8th Pass': 1, '10th Pass (SSC)': 2,
            '12th Pass (HSC)': 3, 'Diploma/ITI': 4, 
            'Graduate (BTech/BA/BCom/BSc)': 5, 'Post Graduate (MBA/MTech/MA/MSc)': 6,
            'PhD/Doctorate': 7
        }
        if 'education' in df.columns:
            df['education_num'] = df['education'].map(edu_map).fillna(3)
            feature_count += 1
        
        print(f"   Added {feature_count} engineered features")
        
        return df
    
    def create_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Create the preprocessing pipeline"""
        print("⚙️ Creating preprocessor...")
        
        # Numeric features (original + engineered)
        numeric_features = [
            'age', 'kids', 'hours_available', 'experience_years',
            'age_squared', 'exp_squared', 'exp_age_ratio',
            'flexibility_score', 'benefits_score', 'city_tier_num',
            'has_kids', 'kids_impact', 'seniority_num',
            'career_growth_num', 'education_num'
        ]
        
        # Original numeric features from config
        for col in self.config.NUMERIC_FEATURES:
            if col not in numeric_features and col in df.columns:
                numeric_features.append(col)
        
        # Filter to only existing columns
        numeric_features = [f for f in numeric_features if f in df.columns]
        
        # Categorical features
        categorical_features = ['domain', 'sector', 'city_tier', 'work_mode', 
                                'work_type', 'shift_type', 'age_group', 'exp_level']
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # Binary features
        binary_features = [f for f in self.config.BINARY_FEATURES if f in df.columns]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                ('bin', 'passthrough', binary_features)
            ],
            remainder='drop'
        )
        
        self.feature_names = numeric_features + categorical_features + binary_features
        
        print(f"   Numeric features: {len(numeric_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        print(f"   Binary features: {len(binary_features)}")
        
        return self.preprocessor
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Full data preparation pipeline"""
        # Clean
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Create preprocessor
        self.create_preprocessor(df)
        
        # Get features
        X = df[self.feature_names]
        
        # Handle text features (TF-IDF + SVD)
        text_features_combined = None
        
        if 'all_skills' in df.columns:
            print("📝 Processing skills text (TF-IDF)...")
            skills_tfidf = self.tfidf_skills.fit_transform(df['all_skills'].fillna(''))
            skills_reduced = self.svd_skills.fit_transform(skills_tfidf)
            text_features_combined = skills_reduced
        
        if 'job_title' in df.columns:
            print("📝 Processing job titles (TF-IDF)...")
            jobs_tfidf = self.tfidf_jobs.fit_transform(df['job_title'].fillna(''))
            jobs_reduced = self.svd_jobs.fit_transform(jobs_tfidf)
            if text_features_combined is not None:
                text_features_combined = np.hstack([text_features_combined, jobs_reduced])
            else:
                text_features_combined = jobs_reduced
        
        # Transform structured features
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Combine with text features
        if text_features_combined is not None:
            X_transformed = np.hstack([X_transformed, text_features_combined])
            print(f"   Combined feature matrix: {X_transformed.shape}")
        
        print(f"   Final feature matrix shape: {X_transformed.shape}")
        
        # Target (income)
        y = df[self.config.TARGET].values
        
        # Log-transform income for better regression
        y_log = np.log1p(y)  # log(1 + y) to handle zeros
        
        print(f"   Target: {self.config.TARGET}")
        print(f"   Applied log-transform to income")
        
        return X_transformed, y_log, df
    
    def inverse_transform_income(self, y_log: np.ndarray) -> np.ndarray:
        """Convert log-transformed predictions back to original scale"""
        return np.expm1(y_log)  # exp(y) - 1
    
    def get_salary_bracket(self, income: float) -> str:
        """Get salary bracket for a given income"""
        for min_val, max_val, bracket in self.config.SALARY_BRACKETS:
            if min_val <= income < max_val:
                return bracket
        return 'Unknown'
    
    def transform_input(self, user_data: Dict) -> np.ndarray:
        """Transform a single user input for prediction"""
        # Create DataFrame
        df = pd.DataFrame([user_data])
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Transform
        X = df[self.feature_names]
        X_transformed = self.preprocessor.transform(X)
        
        # Handle text features
        text_features = []
        
        if 'all_skills' in user_data:
            skills_tfidf = self.tfidf_skills.transform([user_data.get('all_skills', '')])
            skills_reduced = self.svd_skills.transform(skills_tfidf)
            text_features.append(skills_reduced)
        
        if 'job_title' in user_data:
            jobs_tfidf = self.tfidf_jobs.transform([user_data.get('job_title', '')])
            jobs_reduced = self.svd_jobs.transform(jobs_tfidf)
            text_features.append(jobs_reduced)
        
        if text_features:
            text_combined = np.hstack(text_features)
            X_transformed = np.hstack([X_transformed, text_combined])
        
        return X_transformed
    
    def save(self) -> None:
        """Save preprocessing objects"""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        
        joblib.dump({
            'preprocessor': self.preprocessor,
            'tfidf_skills': self.tfidf_skills,
            'tfidf_jobs': self.tfidf_jobs,
            'svd_skills': self.svd_skills,
            'svd_jobs': self.svd_jobs,
            'feature_names': self.feature_names,
            'income_stats': self.income_stats
        }, self.config.PREPROCESSOR_PATH)
        
        print(f"💾 Preprocessor saved to: {self.config.PREPROCESSOR_PATH}")
    
    def load(self) -> None:
        """Load preprocessing objects"""
        data = joblib.load(self.config.PREPROCESSOR_PATH)
        self.preprocessor = data['preprocessor']
        self.tfidf_skills = data['tfidf_skills']
        self.tfidf_jobs = data['tfidf_jobs']
        self.svd_skills = data['svd_skills']
        self.svd_jobs = data['svd_jobs']
        self.feature_names = data['feature_names']
        self.income_stats = data['income_stats']


# ============================================================================
# MODEL TRAINER
# ============================================================================

class IncomePredictionTrainer:
    """Handles model training with multiple algorithms and ensemble"""
    
    def __init__(self, config: IncomeModelConfig = None):
        self.config = config or IncomeModelConfig()
        self.models = {}
        self.ensemble_model = None
        self.metrics = {}
        
    def build_base_models(self) -> Dict[str, Any]:
        """Build all base regression models"""
        print("\n🏗️ Building base models...")
        
        models = {}
        
        # 1. Random Forest Regressor
        models['random_forest'] = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=self.config.N_JOBS,
            random_state=self.config.RANDOM_STATE,
            verbose=0
        )
        print("   ✓ Random Forest Regressor configured")
        
        # 2. Extra Trees Regressor
        models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=self.config.N_JOBS,
            random_state=self.config.RANDOM_STATE,
            verbose=0
        )
        print("   ✓ Extra Trees Regressor configured")
        
        # 3. XGBoost Regressor
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective='reg:squarederror',
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                verbosity=0
            )
            print("   ✓ XGBoost Regressor configured")
        
        # 4. LightGBM Regressor
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=15,
                learning_rate=0.1,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=self.config.N_JOBS,
                random_state=self.config.RANDOM_STATE,
                verbose=-1
            )
            print("   ✓ LightGBM Regressor configured")
        
        # 5. Neural Network (MLP)
        models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=self.config.RANDOM_STATE,
            verbose=False
        )
        print("   ✓ Neural Network (MLP) Regressor configured")
        
        # 6. Gradient Boosting
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            loss='huber',  # More robust to outliers
            random_state=self.config.RANDOM_STATE,
            verbose=0
        )
        print("   ✓ Gradient Boosting Regressor configured")
        
        # 7. Huber Regressor (robust to outliers)
        models['huber'] = HuberRegressor(
            epsilon=1.35,
            max_iter=200,
            alpha=0.0001
        )
        print("   ✓ Huber Regressor configured")
        
        return models
    
    def build_ensemble(self, models: Dict[str, Any]) -> StackingRegressor:
        """Build stacking ensemble from base models"""
        print("\n🔗 Building Stacking Ensemble...")
        
        # Prepare estimators
        estimators = [(name, model) for name, model in models.items()]
        
        # Meta-learner
        meta_learner = Ridge(
            alpha=1.0,
            random_state=self.config.RANDOM_STATE
        )
        
        # Stacking Regressor
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=self.config.N_JOBS,
            verbose=0
        )
        
        print(f"   Stacking ensemble with {len(estimators)} base models")
        
        return ensemble
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        # Convert from log scale to original
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
        
        # Ensure no negative predictions
        y_pred_orig = np.maximum(y_pred_orig, 0)
        
        metrics = {
            f'{prefix}mae': mean_absolute_error(y_true_orig, y_pred_orig),
            f'{prefix}rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
            f'{prefix}r2': r2_score(y_true, y_pred),  # R² on log scale
            f'{prefix}median_ae': median_absolute_error(y_true_orig, y_pred_orig),
            f'{prefix}explained_var': explained_variance_score(y_true, y_pred)
        }
        
        # MAPE (handle division by zero)
        mask = y_true_orig > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100
            metrics[f'{prefix}mape'] = mape
        
        return metrics
    
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
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Training time
            train_time = (datetime.now() - start_time).total_seconds()
            metrics['train_time'] = train_time
            
            results[name] = metrics
            
            print(f"   MAE: ₹{metrics['mae']:,.0f}")
            print(f"   RMSE: ₹{metrics['rmse']:,.0f}")
            print(f"   R²: {metrics['r2']:.4f}")
            if 'mape' in metrics:
                print(f"   MAPE: {metrics['mape']:.1f}%")
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
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        train_time = (datetime.now() - start_time).total_seconds()
        metrics['train_time'] = train_time
        
        print(f"\n   ✅ Ensemble MAE: ₹{metrics['mae']:,.0f}")
        print(f"   ✅ Ensemble RMSE: ₹{metrics['rmse']:,.0f}")
        print(f"   ✅ Ensemble R²: {metrics['r2']:.4f}")
        if 'mape' in metrics:
            print(f"   ✅ Ensemble MAPE: {metrics['mape']:.1f}%")
        print(f"   Training time: {train_time:.1f}s")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Perform cross-validation on all models"""
        print("\n" + "="*70)
        print(f"CROSS-VALIDATION ({self.config.CV_FOLDS}-FOLD)")
        print("="*70)
        
        cv = KFold(
            n_splits=self.config.CV_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE
        )
        
        cv_results = {}
        
        # Limited models for CV (faster)
        cv_models = {k: v for k, v in self.models.items() 
                     if k in ['random_forest', 'xgboost', 'lightgbm']}
        
        for name, model in cv_models.items():
            print(f"\n📊 Cross-validating {name}...")
            
            # R² scores
            r2_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='r2',
                n_jobs=self.config.N_JOBS
            )
            
            # Negative MAE (sklearn convention)
            mae_scores = -cross_val_score(
                model, X, y,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=self.config.N_JOBS
            )
            
            # Convert MAE from log scale to original
            mae_orig = np.expm1(mae_scores) if mae_scores.mean() < 10 else mae_scores
            
            cv_results[name] = {
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std(),
                'mae_mean': mae_orig.mean(),
                'mae_std': mae_orig.std()
            }
            
            print(f"   R² Mean: {r2_scores.mean():.4f} (+/- {r2_scores.std()*2:.4f})")
            print(f"   MAE Mean: ₹{mae_orig.mean():,.0f} (+/- ₹{mae_orig.std():,.0f})")
        
        return cv_results
    
    def save_model(self, model: Any) -> None:
        """Save the trained model"""
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        
        joblib.dump(model, self.config.MODEL_PATH)
        print(f"\n💾 Model saved to: {self.config.MODEL_PATH}")
    
    def save_metadata(self, metrics: Dict, income_stats: Dict) -> None:
        """Save model training metadata"""
        metadata = {
            'model_type': 'Income Prediction',
            'version': '2.0.0',
            'trained_date': datetime.now().isoformat(),
            'algorithm': 'Stacking Ensemble (RF + ET + XGB + LightGBM + MLP + GB + Huber)',
            'cv_folds': self.config.CV_FOLDS,
            'income_stats': income_stats,
            'metrics': metrics
        }
        
        with open(self.config.METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadata saved to: {self.config.METADATA_PATH}")


# ============================================================================
# INCOME PREDICTOR (INFERENCE)
# ============================================================================

class IncomePredictor:
    """Production-ready income prediction engine"""
    
    def __init__(self, config: IncomeModelConfig = None):
        self.config = config or IncomeModelConfig()
        self.model = None
        self.processor = IncomeDataProcessor(config)
        self.is_loaded = False
        
    def load(self) -> bool:
        """Load trained model and preprocessor"""
        try:
            print("🔄 Loading Income Prediction Model...")
            
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
    
    def predict(
        self,
        user_data: Dict,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Predict income for a user
        
        Args:
            user_data: Dictionary with user profile data
            return_confidence: Whether to return confidence interval
            
        Returns:
            Dictionary with predicted income and related info
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Transform input
        X = self.processor.transform_input(user_data)
        
        # Predict (log scale)
        y_log_pred = self.model.predict(X)[0]
        
        # Convert to original scale
        income_pred = np.expm1(y_log_pred)
        income_pred = max(self.config.INCOME_MIN, income_pred)  # Ensure minimum
        income_pred = min(self.config.INCOME_MAX, income_pred)  # Ensure maximum
        
        # Get salary bracket
        salary_bracket = self.processor.get_salary_bracket(income_pred)
        
        # Build result
        result = {
            'predicted_income': round(income_pred),
            'predicted_income_formatted': f"₹{income_pred:,.0f}",
            'salary_bracket': salary_bracket,
            'salary_min': round(income_pred * 0.85),
            'salary_max': round(income_pred * 1.15),
            'salary_range': f"₹{income_pred * 0.85:,.0f} - ₹{income_pred * 1.15:,.0f}"
        }
        
        # Add confidence interval if requested
        if return_confidence:
            # Estimate confidence based on income stats
            std = self.processor.income_stats.get('std', income_pred * 0.2)
            result['confidence_interval'] = {
                'lower': round(max(0, income_pred - std)),
                'upper': round(income_pred + std),
                'confidence_level': '68%'
            }
        
        return result
    
    def batch_predict(self, users_data: List[Dict]) -> List[Dict]:
        """Predict income for multiple users"""
        return [self.predict(user) for user in users_data]
    
    def predict_by_role(
        self,
        job_title: str,
        experience_years: int,
        city_tier: str = 'Tier-1',
        education: str = 'Graduate (BTech/BA/BCom/BSc)'
    ) -> Dict[str, Any]:
        """Quick prediction by job role"""
        user_data = {
            'job_title': job_title,
            'experience_years': experience_years,
            'city_tier': city_tier,
            'education': education,
            'age': 22 + experience_years,
            'kids': 0,
            'hours_available': 8,
            'work_mode': 'Office',
            'domain': 'General',
            'primary_skill': '',
            'all_skills': job_title,
            'marital_status': 'Single',
            'seniority_level': 'Associate' if experience_years < 3 else 'Senior Associate',
            'remote_available': 0,
            'flexible_timing': 0
        }
        return self.predict(user_data)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_income_prediction_model(
    sample_frac: float = None,
    skip_ensemble: bool = False
) -> Dict[str, Any]:
    """
    Main training pipeline for Income Prediction Model
    
    Args:
        sample_frac: Fraction of data to use (None = full dataset)
        skip_ensemble: If True, skip ensemble training (faster)
        
    Returns:
        Dictionary with training results and metrics
    """
    print("\n" + "="*70)
    print("💰 MAASARTHI INCOME PREDICTION MODEL TRAINING")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    config = IncomeModelConfig()
    
    # Step 1: Data Processing
    print("\n" + "-"*50)
    print("STEP 1: DATA PROCESSING")
    print("-"*50)
    
    processor = IncomeDataProcessor(config)
    df = processor.load_data(sample_frac)
    X, y, df_processed = processor.prepare_data(df)
    
    # Step 2: Train-Test Split
    print("\n" + "-"*50)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("-"*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    
    # Step 3: Build Models
    print("\n" + "-"*50)
    print("STEP 3: MODEL BUILDING")
    print("-"*50)
    
    trainer = IncomePredictionTrainer(config)
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
    
    # Step 6: Train Ensemble
    if not skip_ensemble:
        trainer.ensemble_model = trainer.build_ensemble(trainer.models)
        ensemble_results = trainer.train_ensemble(
            X_train, y_train, X_test, y_test
        )
        final_model = trainer.ensemble_model
        final_metrics = ensemble_results
    else:
        # Use best individual model (by R²)
        best_name = max(individual_results.keys(), 
                        key=lambda k: individual_results[k]['r2'])
        print(f"\n🏆 Best individual model: {best_name}")
        final_model = trainer.models[best_name]
        final_metrics = individual_results[best_name]
    
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
    trainer.save_metadata(all_metrics, processor.income_stats)
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Final Model Performance:")
    print(f"   - MAE: ₹{final_metrics['mae']:,.0f}")
    print(f"   - RMSE: ₹{final_metrics['rmse']:,.0f}")
    print(f"   - R²: {final_metrics['r2']:.4f}")
    if 'mape' in final_metrics:
        print(f"   - MAPE: {final_metrics['mape']:.1f}%")
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
    
    parser = argparse.ArgumentParser(description='Train Income Prediction Model')
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
        os.system('pip install xgboost lightgbm --quiet')
    
    # Run training
    results = train_income_prediction_model(
        sample_frac=args.sample,
        skip_ensemble=args.skip_ensemble
    )
