"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 11: MISSING DATA HANDLING
=================================

This script handles missing values in the deduplicated dataset using
intelligent strategies based on column types and business logic.

Key Operations:
- Strategic missing value imputation
- Remove columns with excessive missing data
- Fill critical columns appropriately
- Prepare dataset for ML training

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

class MissingDataHandler:
    def __init__(self, deduplicated_data_path, complete_data_path):
        self.deduplicated_data_path = Path(deduplicated_data_path)
        self.complete_data_path = Path(complete_data_path)
        self.complete_data_path.mkdir(exist_ok=True)
        
        # Thresholds for missing data handling
        self.drop_threshold = 80  # Drop columns with >80% missing data
        self.critical_columns = [
            'job_title', 'company', 'company_name', 'location', 
            'job_category', 'seniority_level'
        ]
        
        # Imputation strategies by column type
        self.imputation_strategies = {
            'categorical_high_frequency': [
                'job_category', 'seniority_level', 'work_arrangement', 
                'city_tier', 'salary_bracket', 'experience_category'
            ],
            'categorical_default': [
                'location', 'source_file', 'data_source'
            ],
            'boolean_false': [
                'remote_flag', 'female_friendly', 'flexible_hours', 
                'part_time_available', 'is_metro', 'is_india', 'leadership_role'
            ],
            'numeric_median': [
                'salary_min', 'salary_max', 'salary_avg', 'required_experience_min', 
                'required_experience_max', 'mother_suitability_score'
            ],
            'numeric_zero': [
                'views', 'applies', 'zip_code', 'fips'
            ]
        }
    
    def analyze_missing_data(self, df):
        """Analyze missing data patterns in the dataset"""
        print("   📊 Analyzing missing data patterns...")
        
        missing_analysis = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            missing_analysis[column] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct,
                'data_type': str(df[column].dtype),
                'unique_values': df[column].nunique(),
                'action_recommended': self._recommend_action(column, missing_pct, df[column])
            }
        
        # Sort by missing percentage
        sorted_missing = sorted(missing_analysis.items(), 
                               key=lambda x: x[1]['missing_percentage'], 
                               reverse=True)
        
        print(f"   📈 Columns with missing data:")
        columns_with_missing = 0
        for col, info in sorted_missing[:10]:  # Show top 10
            if info['missing_percentage'] > 0:
                print(f"      {col:<30} {info['missing_percentage']:>6.1f}% missing ({info['action_recommended']})")
                columns_with_missing += 1
        
        if columns_with_missing == 0:
            print(f"      ✅ No missing data detected!")
        
        return missing_analysis
    
    def _recommend_action(self, column, missing_pct, series):
        """Recommend action for handling missing values"""
        if missing_pct >= self.drop_threshold:
            return 'DROP'
        elif column in self.critical_columns:
            return 'DROP_ROWS'
        elif missing_pct == 0:
            return 'NONE'
        elif series.dtype == 'object':
            return 'FILL_MODE'
        elif series.dtype in ['bool']:
            return 'FILL_FALSE'
        elif series.dtype in ['int64', 'float64']:
            return 'FILL_MEDIAN'
        else:
            return 'FILL_DEFAULT'
    
    def drop_excessive_missing_columns(self, df):
        """Drop columns with excessive missing data"""
        print("   🗑️  Dropping columns with excessive missing data...")
        
        initial_columns = len(df.columns)
        columns_to_drop = []
        
        for column in df.columns:
            missing_pct = (df[column].isnull().sum() / len(df)) * 100
            
            if missing_pct >= self.drop_threshold:
                columns_to_drop.append(column)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"      Dropped {len(columns_to_drop)} columns: {columns_to_drop[:5]}{'...' if len(columns_to_drop) > 5 else ''}")
        
        final_columns = len(df.columns)
        print(f"   ✅ Columns: {initial_columns} → {final_columns}")
        
        return df, columns_to_drop
    
    def handle_critical_missing_rows(self, df):
        """Drop rows missing critical information"""
        print("   🗑️  Handling rows with missing critical data...")
        
        initial_rows = len(df)
        rows_dropped = 0
        
        for column in self.critical_columns:
            if column in df.columns:
                missing_before = df[column].isnull().sum()
                df = df.dropna(subset=[column])
                missing_after = initial_rows - len(df) - rows_dropped
                
                if missing_after > 0:
                    print(f"      Dropped {missing_after} rows missing '{column}'")
                    rows_dropped += missing_after
        
        final_rows = len(df)
        print(f"   ✅ Rows: {initial_rows:,} → {final_rows:,} (-{rows_dropped:,})")
        
        return df, rows_dropped
    
    def impute_missing_values(self, df):
        """Impute missing values using appropriate strategies"""
        print("   🔧 Imputing missing values...")
        
        imputation_summary = {}
        
        # Strategy 1: Categorical - Fill with mode (most frequent value)
        for column in self.imputation_strategies['categorical_high_frequency']:
            if column in df.columns and df[column].isnull().any():
                mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
                missing_count = df[column].isnull().sum()
                df[column].fillna(mode_value, inplace=True)
                imputation_summary[column] = f"Filled {missing_count} with mode: {mode_value}"
        
        # Strategy 2: Categorical - Fill with default values
        categorical_defaults = {
            'location': 'Unknown Location',
            'source_file': 'Unknown Source', 
            'data_source': 'Mixed',
            'job_category': 'Other',
            'seniority_level': 'Mid-Level',
            'work_arrangement': 'On-site',
            'city_tier': 'Unknown',
            'salary_bracket': 'Not Disclosed',
            'experience_category': 'Mid Level'
        }
        
        for column in self.imputation_strategies['categorical_default']:
            if column in df.columns and df[column].isnull().any():
                default_value = categorical_defaults.get(column, 'Not Specified')
                missing_count = df[column].isnull().sum()
                df[column].fillna(default_value, inplace=True)
                imputation_summary[column] = f"Filled {missing_count} with default: {default_value}"
        
        # Strategy 3: Boolean - Fill with False
        for column in self.imputation_strategies['boolean_false']:
            if column in df.columns and df[column].isnull().any():
                missing_count = df[column].isnull().sum()
                df[column].fillna(False, inplace=True)
                imputation_summary[column] = f"Filled {missing_count} with False"
        
        # Strategy 4: Numeric - Fill with median
        for column in self.imputation_strategies['numeric_median']:
            if column in df.columns and df[column].isnull().any():
                median_value = df[column].median()
                missing_count = df[column].isnull().sum()
                if pd.notna(median_value):
                    df[column].fillna(median_value, inplace=True)
                    imputation_summary[column] = f"Filled {missing_count} with median: {median_value:.2f}"
        
        # Strategy 5: Numeric - Fill with zero
        for column in self.imputation_strategies['numeric_zero']:
            if column in df.columns and df[column].isnull().any():
                missing_count = df[column].isnull().sum()
                df[column].fillna(0, inplace=True)
                imputation_summary[column] = f"Filled {missing_count} with 0"
        
        # Strategy 6: Handle remaining missing values
        remaining_missing = df.isnull().sum()
        for column, missing_count in remaining_missing.items():
            if missing_count > 0:
                if df[column].dtype == 'object':
                    df[column].fillna('Not Available', inplace=True)
                    imputation_summary[column] = f"Filled {missing_count} with 'Not Available'"
                else:
                    median_val = df[column].median()
                    fill_val = median_val if pd.notna(median_val) else 0
                    df[column].fillna(fill_val, inplace=True)
                    imputation_summary[column] = f"Filled {missing_count} with {fill_val}"
        
        # Show imputation summary
        if imputation_summary:
            print(f"      Applied {len(imputation_summary)} imputation strategies")
            for col, action in list(imputation_summary.items())[:5]:  # Show first 5
                print(f"      {col}: {action}")
            if len(imputation_summary) > 5:
                print(f"      ... and {len(imputation_summary) - 5} more")
        
        return df, imputation_summary
    
    def validate_completeness(self, df):
        """Validate that dataset is complete after imputation"""
        print("   ✅ Validating dataset completeness...")
        
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        print(f"      Dataset completeness: {completeness:.2f}%")
        
        if missing_cells == 0:
            print(f"      ✅ Dataset is 100% complete!")
            return True
        else:
            print(f"      ⚠️  {missing_cells} missing values remain")
            
            # Show remaining missing columns
            remaining_missing = df.isnull().sum()
            missing_cols = remaining_missing[remaining_missing > 0]
            if len(missing_cols) > 0:
                print(f"      Missing data in: {list(missing_cols.index)[:5]}")
            
            return False
    
    def process_single_dataset(self, filename):
        """Handle missing data in a single dataset"""
        print(f"\n📋 Processing missing data: {filename}")
        
        try:
            # Load deduplicated data
            filepath = self.deduplicated_data_path / filename
            if not filepath.exists():
                print(f"   ❌ File not found: {filename}")
                return None
            
            df = pd.read_csv(filepath, low_memory=False)
            initial_shape = df.shape
            
            # Analyze missing data
            missing_analysis = self.analyze_missing_data(df)
            
            # Step 1: Drop columns with excessive missing data
            df, dropped_columns = self.drop_excessive_missing_columns(df)
            
            # Step 2: Handle rows missing critical information
            df, dropped_rows = self.handle_critical_missing_rows(df)
            
            # Step 3: Impute remaining missing values
            df, imputation_summary = self.impute_missing_values(df)
            
            # Step 4: Validate completeness
            is_complete = self.validate_completeness(df)
            
            # Add processing metadata
            df['missing_data_processed'] = '2026-03-04'
            df['dropped_columns_count'] = len(dropped_columns) if dropped_columns else 0
            df['dropped_rows_count'] = dropped_rows
            df['is_complete_dataset'] = is_complete
            
            # Save complete dataset
            output_filename = filename.replace('deduplicated_', 'complete_')
            output_path = self.complete_data_path / output_filename
            df.to_csv(output_path, index=False)
            
            final_shape = df.shape
            
            print(f"   ✅ Shape: {initial_shape} → {final_shape}")
            print(f"   ✅ Saved: {output_filename}")
            
            return {
                'success': True,
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'dropped_columns': len(dropped_columns) if dropped_columns else 0,
                'dropped_rows': dropped_rows,
                'imputation_summary': imputation_summary,
                'is_complete': is_complete,
                'output_file': output_filename
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_all_datasets(self):
        """Process all deduplicated datasets"""
        print("📋 PHASE 11: MISSING DATA HANDLING")
        print("="*50)
        
        dedup_files = [f for f in os.listdir(self.deduplicated_data_path) 
                      if f.endswith('.csv') and not f.startswith('deduplication_report')]
        
        if not dedup_files:
            print("❌ No deduplicated datasets found!")
            return {}
        
        missing_data_report = {}
        
        for filename in dedup_files:
            result = self.process_single_dataset(filename)
            if result:
                missing_data_report[filename] = result
        
        # Save missing data handling report
        report_path = self.complete_data_path / 'missing_data_handling_report.json'
        with open(report_path, 'w') as f:
            json.dump(missing_data_report, f, indent=2, default=str)
        
        self.generate_summary(missing_data_report)
        return missing_data_report
    
    def generate_summary(self, report):
        """Generate missing data handling summary"""
        print(f"\n📊 MISSING DATA HANDLING SUMMARY")
        print("="*50)
        
        successful = len([k for k, v in report.items() if v.get('success', False)])
        failed = len([k for k, v in report.items() if not v.get('success', False)])
        
        print(f"✅ Successfully processed: {successful} datasets")
        print(f"❌ Failed to process: {failed} datasets")
        
        if successful > 0:
            total_dropped_cols = sum(v.get('dropped_columns', 0) for v in report.values() if v.get('success'))
            total_dropped_rows = sum(v.get('dropped_rows', 0) for v in report.values() if v.get('success'))
            complete_datasets = len([v for v in report.values() if v.get('success') and v.get('is_complete')])
            
            print(f"🗑️  Total columns dropped: {total_dropped_cols}")
            print(f"🗑️  Total rows dropped: {total_dropped_rows:,}")
            print(f"✅ Complete datasets: {complete_datasets}/{successful}")
        
        print(f"\n🎯 READY FOR PHASE 12: Dataset validation")

def main():
    """Main execution function"""
    deduplicated_data_path = Path(__file__).parent.parent / 'Deduplicated_Data'
    complete_data_path = Path(__file__).parent.parent / 'Complete_Data'
    
    if not deduplicated_data_path.exists():
        print("❌ Deduplicated_Data directory not found!")
        print("➡️  Please run phase_10_duplicate_removal.py first")
        return
    
    handler = MissingDataHandler(deduplicated_data_path, complete_data_path)
    report = handler.process_all_datasets()
    
    print("\n🎉 Phase 11 Complete: Missing data handling finished!")
    print("➡️  Next: Run phase_12_dataset_validation.py")

if __name__ == "__main__":
    main()