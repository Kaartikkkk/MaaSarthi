"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 9: DATASET MERGING
=================================

This script combines all cleaned and featured datasets into one master dataset
for MaaSarthi ML training.

Key Operations:
- Identify mergeable datasets (job-related data)
- Standardize column schemas across datasets
- Concatenate datasets vertically
- Handle column mismatches gracefully
- Create unified master dataset

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

class DatasetMerger:
    def __init__(self, feature_data_path, merged_data_path):
        self.feature_data_path = Path(feature_data_path)
        self.merged_data_path = Path(merged_data_path)
        self.merged_data_path.mkdir(exist_ok=True)
        
        # Define which datasets should be merged together
        self.merge_groups = {
            'job_data': [
                'featured_postings.csv',
                'featured_glassdoor_jobs.csv', 
                'featured_salary_data_cleaned.csv',
                'featured_eda_data.csv'
            ],
            'company_data': [
                'featured_companies.csv',
                'featured_companies_india.csv'
            ],
            'skills_data': [
                'featured_job_skills.csv',
                'featured_skills.csv'
            ],
            'reference_data': [
                # Keep separate: benefits, industries, etc.
            ]
        }
        
        # Priority columns for the master dataset
        self.priority_columns = [
            # Core job information
            'job_title', 'company', 'company_name', 'location', 'job_description',
            
            # Salary information  
            'salary_min', 'salary_max', 'salary_avg', 'salary_bracket',
            
            # Experience requirements
            'required_experience_min', 'required_experience_max', 'experience_category',
            
            # New features
            'remote_flag', 'work_arrangement', 'job_category', 'seniority_level',
            'city_tier', 'is_india', 'female_friendly', 'mother_suitability_score',
            
            # Source tracking
            'source_file', 'data_source', 'processing_date'
        ]
    
    def standardize_columns_across_datasets(self, datasets):
        """Ensure all datasets have compatible column structure"""
        print("   🔧 Standardizing columns across datasets...")
        
        # Collect all unique columns
        all_columns = set()
        for df in datasets:
            all_columns.update(df.columns)
        
        # Create column mapping for similar columns
        column_mappings = {
            'company_name': 'company',
            'title': 'job_title', 
            'position': 'job_title',
            'role': 'job_title',
            'city': 'location',
            'description': 'job_description',
            'desc': 'job_description'
        }
        
        standardized_datasets = []
        
        for i, df in enumerate(datasets):
            df_copy = df.copy()
            
            # Apply column mappings
            for old_col, new_col in column_mappings.items():
                if old_col in df_copy.columns and new_col not in df_copy.columns:
                    df_copy.rename(columns={old_col: new_col}, inplace=True)
            
            # Add missing columns with default values
            for col in all_columns:
                if col not in df_copy.columns:
                    # Determine appropriate default value
                    if any(keyword in col.lower() for keyword in ['flag', 'is_']):
                        df_copy[col] = False
                    elif any(keyword in col.lower() for keyword in ['score', 'rating', 'min', 'max']):
                        df_copy[col] = np.nan
                    else:
                        df_copy[col] = 'Not Specified'
            
            # Reorder columns to match first dataset
            if i == 0:
                column_order = list(df_copy.columns)
            else:
                # Reorder to match the first dataset's column order
                missing_cols = set(df_copy.columns) - set(column_order)
                df_copy = df_copy[column_order + list(missing_cols)]
            
            standardized_datasets.append(df_copy)
        
        return standardized_datasets, column_order
    
    def merge_dataset_group(self, group_name, filenames):
        """Merge a group of related datasets"""
        print(f"\n📦 Merging {group_name}...")
        
        datasets_to_merge = []
        dataset_info = []
        
        # Load each dataset in the group
        for filename in filenames:
            filepath = self.feature_data_path / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath, low_memory=False)
                    datasets_to_merge.append(df)
                    dataset_info.append({
                        'filename': filename,
                        'rows': len(df),
                        'columns': len(df.columns)
                    })
                    print(f"   📊 Loaded {filename}: {len(df):,} rows")
                except Exception as e:
                    print(f"   ⚠️  Could not load {filename}: {e}")
            else:
                print(f"   ⚠️  File not found: {filename}")
        
        if not datasets_to_merge:
            print(f"   ❌ No datasets to merge for {group_name}")
            return None, []
        
        if len(datasets_to_merge) == 1:
            print(f"   ℹ️  Only one dataset in group, using as-is")
            return datasets_to_merge[0], dataset_info
        
        # Standardize columns across datasets
        standardized_datasets, column_order = self.standardize_columns_across_datasets(datasets_to_merge)
        
        # Merge datasets
        print(f"   🔗 Concatenating {len(standardized_datasets)} datasets...")
        
        merged_df = pd.concat(standardized_datasets, ignore_index=True, sort=False)
        
        # Add merge metadata
        merged_df['merge_group'] = group_name
        merged_df['original_dataset_count'] = len(datasets_to_merge)
        
        print(f"   ✅ Merged dataset: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
        
        return merged_df, dataset_info
    
    def create_master_dataset(self):
        """Create the main MaaSarthi master dataset from job-related data"""
        print(f"\n🎯 Creating MaaSarthi Master Dataset...")
        
        # Merge job data (main dataset)
        job_df, job_info = self.merge_dataset_group('job_data', self.merge_groups['job_data'])
        
        if job_df is None:
            print("   ❌ Could not create master dataset - no job data available")
            return None
        
        # Enhance with company data if available
        company_df, company_info = self.merge_dataset_group('company_data', self.merge_groups['company_data'])
        
        if company_df is not None:
            # Try to merge company information
            print("   🔗 Enriching with company data...")
            
            # Find common columns for merging
            common_cols = []
            for col in ['company', 'company_name', 'company_id']:
                if col in job_df.columns and col in company_df.columns:
                    common_cols.append(col)
            
            if common_cols:
                merge_col = common_cols[0]
                print(f"   📊 Merging on column: {merge_col}")
                
                # Add prefix to company columns to avoid conflicts
                company_cols_renamed = {}
                for col in company_df.columns:
                    if col != merge_col and col in job_df.columns:
                        company_cols_renamed[col] = f"company_{col}"
                
                if company_cols_renamed:
                    company_df = company_df.rename(columns=company_cols_renamed)
                
                # Perform left join to enrich job data
                initial_rows = len(job_df)
                job_df = pd.merge(job_df, company_df, on=merge_col, how='left')
                
                print(f"   ✅ Enhanced {len(job_df)} rows with company data")
            else:
                print("   ⚠️  No common columns found for company merge")
        
        # Add final processing metadata
        job_df['master_dataset_version'] = '1.0'
        job_df['created_date'] = '2026-03-04'
        job_df['record_id'] = range(1, len(job_df) + 1)
        
        # Prioritize important columns
        priority_cols_present = [col for col in self.priority_columns if col in job_df.columns]
        other_cols = [col for col in job_df.columns if col not in priority_cols_present]
        
        # Reorder columns with priority columns first
        final_column_order = priority_cols_present + other_cols
        job_df = job_df[final_column_order]
        
        return job_df, {'job_data': job_info, 'company_data': company_info}
    
    def save_master_dataset(self, master_df, metadata):
        """Save the master dataset and create summary report"""
        
        if master_df is None:
            print("❌ No master dataset to save")
            return
        
        # Save master dataset
        master_file = self.merged_data_path / 'maasarthi_master_dataset.csv'
        master_df.to_csv(master_file, index=False)
        
        print(f"💾 Master dataset saved: {master_file}")
        print(f"📊 Final shape: {master_df.shape}")
        
        # Create and save metadata report
        report = {
            'creation_date': '2026-03-04',
            'total_rows': len(master_df),
            'total_columns': len(master_df.columns),
            'source_datasets': metadata,
            'column_summary': {
                'priority_columns': len([col for col in self.priority_columns if col in master_df.columns]),
                'total_columns': len(master_df.columns),
                'data_types': master_df.dtypes.astype(str).to_dict()
            },
            'data_quality': {
                'missing_values': master_df.isnull().sum().to_dict(),
                'duplicate_rows': master_df.duplicated().sum(),
                'unique_companies': master_df['company'].nunique() if 'company' in master_df.columns else 0,
                'india_jobs': master_df['is_india'].sum() if 'is_india' in master_df.columns else 0,
                'remote_jobs': master_df['remote_flag'].sum() if 'remote_flag' in master_df.columns else 0
            }
        }
        
        # Save metadata
        metadata_file = self.merged_data_path / 'master_dataset_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def process_merge_operation(self):
        """Execute the complete merge process"""
        print("📦 PHASE 9: DATASET MERGING")
        print("="*50)
        
        # Create master dataset
        master_df, metadata = self.create_master_dataset()
        
        # Save all individual merged groups
        for group_name, filenames in self.merge_groups.items():
            if filenames:  # Only process non-empty groups
                merged_df, group_info = self.merge_dataset_group(group_name, filenames)
                
                if merged_df is not None:
                    output_file = self.merged_data_path / f'merged_{group_name}.csv'
                    merged_df.to_csv(output_file, index=False)
                    print(f"   💾 Saved: {output_file}")
        
        # Save master dataset
        report = self.save_master_dataset(master_df, metadata)
        
        self.generate_summary(report)
        
        return master_df, report
    
    def generate_summary(self, report):
        """Generate merge summary"""
        print(f"\n📊 DATASET MERGING SUMMARY")
        print("="*50)
        
        if report:
            print(f"✅ Master dataset created successfully!")
            print(f"📊 Total rows: {report['total_rows']:,}")
            print(f"📋 Total columns: {report['total_columns']}")
            
            quality = report['data_quality']
            print(f"🏢 Unique companies: {quality['unique_companies']:,}")
            print(f"🇮🇳 India jobs: {quality['india_jobs']:,}")
            print(f"🏠 Remote jobs: {quality['remote_jobs']:,}")
            print(f"🔄 Duplicate rows: {quality['duplicate_rows']:,}")
            
            # Show missing value summary
            missing_counts = quality['missing_values']
            columns_with_missing = {k: v for k, v in missing_counts.items() if v > 0}
            
            if columns_with_missing:
                print(f"\n⚠️  Columns with missing values: {len(columns_with_missing)}")
            else:
                print(f"\n✅ No missing values detected!")
        
        print(f"\n🎯 READY FOR PHASE 10: Duplicate removal")

def main():
    """Main execution function"""
    feature_data_path = Path(__file__).parent.parent / 'Feature_Data'
    merged_data_path = Path(__file__).parent.parent / 'Merged_Data'
    
    if not feature_data_path.exists():
        print("❌ Feature_Data directory not found!")
        print("➡️  Please run phase_08_feature_creation.py first")
        return
    
    merger = DatasetMerger(feature_data_path, merged_data_path)
    master_df, report = merger.process_merge_operation()
    
    print("\n🎉 Phase 9 Complete: Dataset merging finished!")
    print("➡️  Next: Run phase_10_duplicate_removal.py")

if __name__ == "__main__":
    main()