"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 4: COLUMN STANDARDIZATION
=================================

This script standardizes column names across all datasets to create
a unified schema for data merging and processing.

Key Operations:
- Rename columns to follow consistent naming convention
- Map similar columns across different datasets
- Create standardized data types
- Handle special characters and spaces

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

class ColumnStandardizer:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(exist_ok=True)
        
        # Standard column mapping for MaaSarthi
        self.column_mapping = {
            # Job-related columns
            'job_title': ['title', 'job_title', 'position', 'role', 'Job Title'],
            'company': ['company', 'company_name', 'employer', 'Company Name'],
            'location': ['location', 'city', 'address', 'Location'],
            'salary': ['salary', 'salary_estimate', 'compensation', 'Salary Estimate'],
            'experience': ['experience', 'exp', 'years_exp', 'experience_required'],
            'skills': ['skills', 'skill', 'required_skills', 'skill_abr'],
            'job_description': ['description', 'job_description', 'desc', 'Job Description'],
            
            # Company-related columns
            'company_id': ['company_id', 'id', 'employer_id'],
            'company_size': ['company_size', 'size', 'employees', 'Size'],
            'industry': ['industry', 'sector', 'domain', 'Industry', 'Sector'],
            'founded': ['founded', 'established', 'Founded'],
            'headquarters': ['headquarters', 'hq', 'Headquarters'],
            
            # Demographics/Gender columns
            'country_code': ['CountryCode', 'country_code', 'country'],
            'series_code': ['SeriesCode', 'series_code', 'indicator'],
            'year': ['year', 'time', 'date', 'period'],
            'value': ['value', 'data', 'count', 'percentage'],
            
            # User profile columns (for synthetic data)
            'user_id': ['user_id', 'id', 'profile_id'],
            'age': ['age', 'years'],
            'education': ['education', 'qualification', 'degree'],
            'gender': ['gender', 'sex'],
        }
        
        # Data type mapping
        self.dtype_mapping = {
            'company_id': 'int64',
            'job_id': 'int64', 
            'user_id': 'str',
            'salary_min': 'float64',
            'salary_max': 'float64',
            'experience_min': 'float64',
            'experience_max': 'float64',
            'age': 'int64',
            'founded': 'int64',
            'year': 'int64',
            'value': 'float64'
        }
        
    def standardize_column_names(self, df, dataset_name):
        """Standardize column names for a single dataset"""
        df_copy = df.copy()
        original_columns = list(df_copy.columns)
        
        # Clean column names - remove spaces, special characters, convert to lowercase
        df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        # Apply specific mapping based on dataset type
        renamed_columns = {}
        
        for standard_name, variations in self.column_mapping.items():
            for col in df_copy.columns:
                if any(var.lower().replace(' ', '_') == col for var in variations):
                    if col not in renamed_columns.values():  # Avoid duplicate mappings
                        renamed_columns[col] = standard_name
                        break
        
        # Apply renaming
        df_copy.rename(columns=renamed_columns, inplace=True)
        
        # Dataset-specific column handling
        if 'job' in dataset_name.lower() or 'posting' in dataset_name.lower():
            df_copy = self._handle_job_dataset_columns(df_copy)
        elif 'company' in dataset_name.lower():
            df_copy = self._handle_company_dataset_columns(df_copy)
        elif 'gender' in dataset_name.lower() or 'stats' in dataset_name.lower():
            df_copy = self._handle_gender_dataset_columns(df_copy)
        elif 'salary' in dataset_name.lower():
            df_copy = self._handle_salary_dataset_columns(df_copy)
        
        return df_copy, original_columns, renamed_columns
    
    def _handle_job_dataset_columns(self, df):
        """Handle job-specific column standardization"""
        # Standardize common job dataset columns
        column_renames = {
            'jobtitle': 'job_title',
            'companyname': 'company',
            'salaryestimate': 'salary_estimate', 
            'jobdescription': 'job_description',
            'rating': 'company_rating',
            'typeofownership': 'ownership_type',
            'revenue': 'company_revenue'
        }
        
        for old_col, new_col in column_renames.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        return df
    
    def _handle_company_dataset_columns(self, df):
        """Handle company-specific column standardization"""
        column_renames = {
            'name': 'company_name',
            'companysize': 'company_size',
            'zipcode': 'zip_code',
            'state': 'state',
            'city': 'city',
            'url': 'company_url'
        }
        
        for old_col, new_col in column_renames.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        return df
    
    def _handle_gender_dataset_columns(self, df):
        """Handle gender statistics column standardization"""
        column_renames = {
            'countrycode': 'country_code',
            'seriescode': 'series_code',
            'description': 'indicator_description'
        }
        
        for old_col, new_col in column_renames.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        return df
    
    def _handle_salary_dataset_columns(self, df):
        """Handle salary dataset column standardization"""
        column_renames = {
            'minsalary': 'salary_min',
            'maxsalary': 'salary_max',
            'avgsalary': 'salary_avg',
            'hourly': 'is_hourly',
            'employerprovided': 'employer_provided'
        }
        
        for old_col, new_col in column_renames.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        return df
    
    def standardize_data_types(self, df):
        """Standardize data types across datasets"""
        for col, dtype in self.dtype_mapping.items():
            if col in df.columns:
                try:
                    if dtype == 'int64':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    elif dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif dtype == 'str':
                        df[col] = df[col].astype(str)
                except Exception as e:
                    print(f"   ⚠️  Could not convert {col} to {dtype}: {e}")
        
        return df
    
    def process_all_datasets(self):
        """Process all datasets in the raw data directory"""
        print("🔄 PHASE 4: COLUMN STANDARDIZATION")
        print("="*50)
        
        csv_files = [f for f in os.listdir(self.raw_data_path) if f.endswith('.csv')]
        standardization_report = {}
        
        for filename in csv_files:
            print(f"\n📋 Processing: {filename}")
            
            try:
                # Load dataset
                df = pd.read_csv(self.raw_data_path / filename, low_memory=False)
                original_shape = df.shape
                
                # Standardize columns
                df_standardized, original_columns, renamed_columns = self.standardize_column_names(df, filename)
                
                # Standardize data types
                df_standardized = self.standardize_data_types(df_standardized)
                
                # Save processed dataset
                output_filename = f"standardized_{filename}"
                output_path = self.processed_data_path / output_filename
                df_standardized.to_csv(output_path, index=False)
                
                # Update report
                standardization_report[filename] = {
                    'original_columns': original_columns,
                    'renamed_columns': renamed_columns,
                    'final_columns': list(df_standardized.columns),
                    'original_shape': original_shape,
                    'final_shape': df_standardized.shape,
                    'output_file': output_filename
                }
                
                print(f"   ✅ Columns: {len(original_columns)} → {len(df_standardized.columns)}")
                print(f"   ✅ Renamed: {len(renamed_columns)} columns")
                print(f"   ✅ Saved: {output_filename}")
                
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")
                standardization_report[filename] = {'error': str(e)}
        
        # Save standardization report
        report_path = self.processed_data_path / 'standardization_report.json'
        with open(report_path, 'w') as f:
            json.dump(standardization_report, f, indent=2, default=str)
        
        self.generate_summary(standardization_report)
        return standardization_report
    
    def generate_summary(self, report):
        """Generate standardization summary"""
        print(f"\n📊 STANDARDIZATION SUMMARY")
        print("="*50)
        
        successful = len([k for k, v in report.items() if 'error' not in v])
        failed = len([k for k, v in report.items() if 'error' in v])
        
        print(f"✅ Successfully processed: {successful} datasets")
        print(f"❌ Failed to process: {failed} datasets")
        
        # Show most renamed columns
        total_renamed = sum(len(v.get('renamed_columns', {})) for v in report.values() if 'renamed_columns' in v)
        print(f"🔄 Total columns renamed: {total_renamed}")
        
        print(f"\n🎯 KEY DATASETS FOR MAASARTHI:")
        maasarthi_datasets = ['postings', 'companies_india', 'job_skills', 'salary_data_cleaned', 'glassdoor_jobs']
        
        for dataset_key in maasarthi_datasets:
            matching_files = [k for k in report.keys() if any(key in k.lower() for key in dataset_key.split('_'))]
            for filename in matching_files:
                if filename in report and 'error' not in report[filename]:
                    info = report[filename]
                    print(f"   ✅ {filename:<30} {info['final_shape'][0]:>6,} rows, {info['final_shape'][1]:>2} cols")
                    break

def main():
    """Main execution function"""
    raw_data_path = Path(__file__).parent.parent / 'Raw_Data'
    processed_data_path = Path(__file__).parent.parent / 'Processed_Data'
    
    if not raw_data_path.exists():
        print("❌ Raw_Data directory not found!")
        return
    
    standardizer = ColumnStandardizer(raw_data_path, processed_data_path)
    report = standardizer.process_all_datasets()
    
    print("\n🎉 Phase 4 Complete: Column standardization finished!")
    print("➡️  Next: Run phase_05_data_cleaning.py")

if __name__ == "__main__":
    main()