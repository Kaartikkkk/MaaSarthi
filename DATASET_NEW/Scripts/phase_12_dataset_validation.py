"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 12: DATASET VALIDATION
=================================

This script performs comprehensive validation of the complete dataset
to ensure data quality and readiness for ML model training.

Key Operations:
- Check dataset size and structure
- Validate data ranges and distributions
- Identify data quality issues
- Generate comprehensive quality report
- Ensure ML readiness

Author: MaaSarthi Data Team
Date: March 2026
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DatasetValidator:
    def __init__(self, complete_data_path, validated_data_path):
        self.complete_data_path = Path(complete_data_path)
        self.validated_data_path = Path(validated_data_path)
        self.validated_data_path.mkdir(exist_ok=True)
        
        # Validation criteria
        self.validation_rules = {
            'minimum_rows': 100,       # Minimum dataset size
            'maximum_rows': 1000000,   # Maximum reasonable size
            'minimum_columns': 10,     # Minimum feature count
            'maximum_duplicates': 0.05, # Max 5% duplicates allowed
            'minimum_completeness': 0.95, # Min 95% completeness
            'salary_range': (10000, 10000000),  # Reasonable salary range (INR)
            'experience_range': (0, 50),        # Reasonable experience range (years)
        }
        
        # Expected column categories
        self.expected_column_types = {
            'identifiers': ['record_id', 'job_id', 'company_id', 'user_id'],
            'job_info': ['job_title', 'company', 'location', 'job_description', 'job_category'],
            'salary': ['salary_min', 'salary_max', 'salary_avg', 'salary_bracket'],
            'experience': ['required_experience_min', 'required_experience_max', 'experience_category'],
            'features': ['remote_flag', 'work_arrangement', 'seniority_level', 'city_tier'],
            'maasarthi_specific': ['female_friendly', 'mother_suitability_score', 'is_india'],
            'metadata': ['source_file', 'data_source', 'processing_date', 'created_date']
        }
    
    def validate_dataset_structure(self, df, filename):
        """Validate basic dataset structure"""
        print("   📏 Validating dataset structure...")
        
        validation_results = {
            'filename': filename,
            'shape': df.shape,
            'size_validation': 'PASS',
            'structure_issues': []
        }
        
        rows, cols = df.shape
        
        # Check dataset size
        if rows < self.validation_rules['minimum_rows']:
            validation_results['size_validation'] = 'FAIL'
            validation_results['structure_issues'].append(f"Dataset too small: {rows} rows (min: {self.validation_rules['minimum_rows']})")
        
        if rows > self.validation_rules['maximum_rows']:
            validation_results['size_validation'] = 'WARNING'
            validation_results['structure_issues'].append(f"Dataset very large: {rows:,} rows")
        
        if cols < self.validation_rules['minimum_columns']:
            validation_results['size_validation'] = 'FAIL'
            validation_results['structure_issues'].append(f"Too few columns: {cols} (min: {self.validation_rules['minimum_columns']})")
        
        # Check for critical columns
        critical_columns = ['job_title', 'company']
        missing_critical = [col for col in critical_columns if col not in df.columns]
        
        if missing_critical:
            validation_results['size_validation'] = 'FAIL'
            validation_results['structure_issues'].append(f"Missing critical columns: {missing_critical}")
        
        print(f"      Dataset shape: {df.shape}")
        print(f"      Size validation: {validation_results['size_validation']}")
        
        return validation_results
    
    def validate_data_quality(self, df):
        """Validate data quality metrics"""
        print("   🔍 Validating data quality...")
        
        quality_results = {
            'completeness': 0,
            'duplicates': 0,
            'data_types': {},
            'quality_score': 0,
            'quality_issues': []
        }
        
        # Check completeness
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        quality_results['completeness'] = completeness
        
        if completeness < self.validation_rules['minimum_completeness']:
            quality_results['quality_issues'].append(f"Low completeness: {completeness:.2%} (min: {self.validation_rules['minimum_completeness']:.1%})")
        
        # Check duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_rate = duplicate_count / len(df)
        quality_results['duplicates'] = duplicate_rate
        
        if duplicate_rate > self.validation_rules['maximum_duplicates']:
            quality_results['quality_issues'].append(f"High duplicate rate: {duplicate_rate:.2%} (max: {self.validation_rules['maximum_duplicates']:.1%})")
        
        # Check data types
        for column in df.columns:
            dtype = str(df[column].dtype)
            quality_results['data_types'][column] = dtype
            
            # Validate specific column types
            if 'salary' in column.lower() and 'bracket' not in column.lower():
                if dtype not in ['int64', 'float64']:
                    quality_results['quality_issues'].append(f"Invalid data type for {column}: {dtype} (should be numeric)")
            
            if 'flag' in column.lower() or column.lower().startswith('is_'):
                if dtype not in ['bool', 'object']:
                    quality_results['quality_issues'].append(f"Invalid data type for {column}: {dtype} (should be boolean)")
        
        # Calculate quality score
        quality_score = (completeness * 50 +  # 50 points for completeness
                        (1 - duplicate_rate) * 30 +  # 30 points for uniqueness  
                        (1 if not quality_results['quality_issues'] else 0.5) * 20)  # 20 points for type consistency
        
        quality_results['quality_score'] = min(quality_score, 100)
        
        print(f"      Completeness: {completeness:.1%}")
        print(f"      Duplicates: {duplicate_rate:.1%}")
        print(f"      Quality score: {quality_score:.1f}/100")
        
        return quality_results
    
    def validate_business_rules(self, df):
        """Validate business logic and data ranges"""
        print("   💼 Validating business rules...")
        
        business_results = {
            'salary_validation': 'PASS',
            'experience_validation': 'PASS', 
            'india_jobs_validation': 'PASS',
            'business_issues': []
        }
        
        # Validate salary ranges
        salary_columns = [col for col in df.columns if 'salary' in col.lower() and col.lower() not in ['salary_bracket', 'salary_category']]
        
        for col in salary_columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                min_sal = df[col].min()
                max_sal = df[col].max()
                
                sal_min, sal_max = self.validation_rules['salary_range']
                
                if pd.notna(min_sal) and min_sal < sal_min:
                    business_results['salary_validation'] = 'WARNING'
                    business_results['business_issues'].append(f"{col} has unrealistic low values: {min_sal}")
                
                if pd.notna(max_sal) and max_sal > sal_max:
                    business_results['salary_validation'] = 'WARNING'
                    business_results['business_issues'].append(f"{col} has unrealistic high values: {max_sal:,.0f}")
        
        # Validate experience ranges
        experience_columns = [col for col in df.columns if 'experience' in col.lower() and 'min' in col.lower() or 'max' in col.lower()]
        
        for col in experience_columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                min_exp = df[col].min()
                max_exp = df[col].max()
                
                exp_min, exp_max = self.validation_rules['experience_range']
                
                if pd.notna(min_exp) and min_exp < exp_min:
                    business_results['experience_validation'] = 'WARNING'
                    business_results['business_issues'].append(f"{col} has negative values: {min_exp}")
                
                if pd.notna(max_exp) and max_exp > exp_max:
                    business_results['experience_validation'] = 'WARNING'
                    business_results['business_issues'].append(f"{col} has unrealistic values: {max_exp}")
        
        # Validate India-specific data
        if 'is_india' in df.columns:
            india_jobs_count = df['is_india'].sum() if 'is_india' in df.columns else 0
            india_percentage = india_jobs_count / len(df) * 100
            
            if india_percentage < 10:  # Less than 10% India jobs might be concerning
                business_results['india_jobs_validation'] = 'WARNING'
                business_results['business_issues'].append(f"Low India job percentage: {india_percentage:.1f}%")
        
        # Validate MaaSarthi-specific features
        if 'mother_suitability_score' in df.columns:
            score_range = df['mother_suitability_score'].describe()
            if score_range['max'] > 10 or score_range['min'] < 0:
                business_results['business_issues'].append("Mother suitability score out of expected range (0-10)")
        
        print(f"      Salary validation: {business_results['salary_validation']}")
        print(f"      Experience validation: {business_results['experience_validation']}")
        print(f"      India jobs validation: {business_results['india_jobs_validation']}")
        
        return business_results
    
    def validate_ml_readiness(self, df):
        """Validate dataset readiness for ML training"""
        print("   🤖 Validating ML readiness...")
        
        ml_results = {
            'feature_count': 0,
            'categorical_features': 0,
            'numerical_features': 0,
            'target_candidates': [],
            'ml_readiness_score': 0,
            'ml_issues': []
        }
        
        # Count feature types
        categorical_cols = []
        numerical_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'float64', 'bool']:
                numerical_cols.append(col)
        
        ml_results['categorical_features'] = len(categorical_cols)
        ml_results['numerical_features'] = len(numerical_cols)
        ml_results['feature_count'] = len(categorical_cols) + len(numerical_cols)
        
        # Identify potential target variables for MaaSarthi
        target_candidates = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['salary', 'suitability', 'category', 'bracket']):
                target_candidates.append(col)
        
        ml_results['target_candidates'] = target_candidates
        
        # Check for high cardinality categorical variables
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count > 1000:
                ml_results['ml_issues'].append(f"High cardinality categorical: {col} ({unique_count} unique values)")
        
        # Check feature balance
        if len(categorical_cols) < 5:
            ml_results['ml_issues'].append("Few categorical features - consider feature engineering")
        
        if len(numerical_cols) < 5:
            ml_results['ml_issues'].append("Few numerical features - consider creating derived features")
        
        # Calculate ML readiness score
        feature_score = min(ml_results['feature_count'] / 20 * 40, 40)  # 40 points for good feature count
        target_score = 20 if target_candidates else 0  # 20 points for having target candidates
        balance_score = 20 if len(categorical_cols) >= 5 and len(numerical_cols) >= 5 else 10
        issue_penalty = len(ml_results['ml_issues']) * 5
        
        ml_results['ml_readiness_score'] = max(feature_score + target_score + balance_score - issue_penalty, 0)
        
        print(f"      Feature count: {ml_results['feature_count']}")
        print(f"      Categorical: {len(categorical_cols)}, Numerical: {len(numerical_cols)}")
        print(f"      Target candidates: {len(target_candidates)}")
        print(f"      ML readiness: {ml_results['ml_readiness_score']:.0f}/100")
        
        return ml_results
    
    def generate_data_profile(self, df):
        """Generate comprehensive data profile"""
        print("   📊 Generating data profile...")
        
        profile = {
            'basic_stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024**2):.1f} MB",
                'data_types': df.dtypes.value_counts().to_dict()
            },
            'numeric_summary': {},
            'categorical_summary': {},
            'missing_data_summary': df.isnull().sum().to_dict()
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            profile['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() < 50:  # Only for low cardinality
                profile['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
        
        return profile
    
    def validate_single_dataset(self, filename):
        """Validate a single complete dataset"""
        print(f"\n✅ Validating: {filename}")
        
        try:
            # Load complete data
            filepath = self.complete_data_path / filename
            if not filepath.exists():
                print(f"   ❌ File not found: {filename}")
                return None
            
            df = pd.read_csv(filepath, low_memory=False)
            
            # Perform all validations
            structure_results = self.validate_dataset_structure(df, filename)
            quality_results = self.validate_data_quality(df)
            business_results = self.validate_business_rules(df)
            ml_results = self.validate_ml_readiness(df)
            data_profile = self.generate_data_profile(df)
            
            # Calculate overall validation score
            structure_score = 25 if structure_results['size_validation'] == 'PASS' else 0
            quality_score = quality_results['quality_score'] * 0.25
            business_score = 25 if not business_results['business_issues'] else 15
            ml_score = ml_results['ml_readiness_score'] * 0.25
            
            overall_score = structure_score + quality_score + business_score + ml_score
            
            # Determine validation status
            if overall_score >= 80:
                validation_status = 'EXCELLENT'
            elif overall_score >= 65:
                validation_status = 'GOOD'
            elif overall_score >= 50:
                validation_status = 'ACCEPTABLE'
            else:
                validation_status = 'NEEDS_IMPROVEMENT'
            
            # Create validated dataset (copy with validation metadata)
            df['validation_date'] = '2026-03-04'
            df['validation_score'] = overall_score
            df['validation_status'] = validation_status
            
            # Save validated dataset
            output_filename = filename.replace('complete_', 'validated_')
            output_path = self.validated_data_path / output_filename
            df.to_csv(output_path, index=False)
            
            print(f"   📊 Overall score: {overall_score:.1f}/100 ({validation_status})")
            print(f"   ✅ Saved: {output_filename}")
            
            return {
                'success': True,
                'validation_status': validation_status,
                'overall_score': overall_score,
                'structure_results': structure_results,
                'quality_results': quality_results,
                'business_results': business_results,
                'ml_results': ml_results,
                'data_profile': data_profile,
                'output_file': output_filename
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_all_datasets(self):
        """Validate all complete datasets"""
        print("✅ PHASE 12: DATASET VALIDATION")
        print("="*50)
        
        complete_files = [f for f in os.listdir(self.complete_data_path) 
                         if f.endswith('.csv') and not f.startswith('missing_data_handling_report')]
        
        if not complete_files:
            print("❌ No complete datasets found!")
            return {}
        
        validation_report = {}
        
        for filename in complete_files:
            result = self.validate_single_dataset(filename)
            if result:
                validation_report[filename] = result
        
        # Save validation report
        report_path = self.validated_data_path / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        self.generate_summary(validation_report)
        return validation_report
    
    def generate_summary(self, report):
        """Generate validation summary"""
        print(f"\n📊 DATASET VALIDATION SUMMARY")
        print("="*50)
        
        successful = len([k for k, v in report.items() if v.get('success', False)])
        failed = len([k for k, v in report.items() if not v.get('success', False)])
        
        print(f"✅ Successfully validated: {successful} datasets")
        print(f"❌ Failed to validate: {failed} datasets")
        
        if successful > 0:
            # Count by validation status
            status_counts = {}
            avg_score = 0
            
            for v in report.values():
                if v.get('success'):
                    status = v.get('validation_status', 'UNKNOWN')
                    status_counts[status] = status_counts.get(status, 0) + 1
                    avg_score += v.get('overall_score', 0)
            
            avg_score /= successful
            
            print(f"📈 Average validation score: {avg_score:.1f}/100")
            print(f"📊 Validation status breakdown:")
            for status, count in status_counts.items():
                print(f"   {status}: {count} datasets")
        
        print(f"\n🎯 READY FOR PHASE 13: Master dataset creation")

def main():
    """Main execution function"""
    complete_data_path = Path(__file__).parent.parent / 'Complete_Data'
    validated_data_path = Path(__file__).parent.parent / 'Validated_Data'
    
    if not complete_data_path.exists():
        print("❌ Complete_Data directory not found!")
        print("➡️  Please run phase_11_missing_data_handling.py first")
        return
    
    validator = DatasetValidator(complete_data_path, validated_data_path)
    report = validator.process_all_datasets()
    
    print("\n🎉 Phase 12 Complete: Dataset validation finished!")
    print("➡️  Next: Run phase_13_master_dataset_creation.py")

if __name__ == "__main__":
    main()