"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 5: DATA CLEANING
=================================

This script cleans text fields, removes duplicates, handles missing values,
and standardizes text formatting across all datasets.

Key Operations:
- Clean job titles, company names, locations
- Remove extra spaces and special characters
- Standardize capitalization
- Handle missing values appropriately
- Remove duplicates

Author: MaaSarthi Data Team  
Date: March 2026
"""

import pandas as pd
import numpy as np
import os
import re
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, processed_data_path, cleaned_data_path):
        self.processed_data_path = Path(processed_data_path)
        self.cleaned_data_path = Path(cleaned_data_path)
        self.cleaned_data_path.mkdir(exist_ok=True)
        
        # Text cleaning patterns
        self.text_patterns = {
            'extra_spaces': r'\s+',
            'special_chars': r'[^\w\s\-\.\,\(\)]',
            'multiple_commas': r',+',
            'leading_trailing': r'^\s+|\s+$'
        }
        
        # Location standardization mappings
        self.location_mapping = {
            # Indian cities standardization
            'bengaluru': 'bangalore',
            'mumbai': 'mumbai', 
            'delhi ncr': 'delhi',
            'new delhi': 'delhi',
            'gurgaon': 'gurugram',
            'noida': 'noida',
            'pune': 'pune',
            'hyderabad': 'hyderabad',
            'chennai': 'chennai',
            'kolkata': 'kolkata',
            'ahmedabad': 'ahmedabad',
            'kochi': 'kochi',
            'coimbatore': 'coimbatore',
            'indore': 'indore',
            'jaipur': 'jaipur',
            'chandigarh': 'chandigarh'
        }
        
        # Company name cleaning patterns
        self.company_patterns = {
            'ltd': ['ltd', 'limited', 'ltd.', 'limited.'],
            'pvt': ['pvt', 'private', 'pvt.', 'private.'],
            'inc': ['inc', 'incorporated', 'inc.'],
            'corp': ['corp', 'corporation', 'corp.'],
            'llp': ['llp', 'limited liability partnership'],
            'llc': ['llc', 'limited liability company']
        }
    
    def clean_text_field(self, series, field_type='general'):
        """Clean and standardize text fields"""
        if series.empty:
            return series
            
        # Convert to string and handle nulls
        cleaned = series.astype(str).replace('nan', '')
        
        # Remove extra spaces
        cleaned = cleaned.str.replace(self.text_patterns['extra_spaces'], ' ', regex=True)
        
        # Remove special characters (keep basic punctuation)
        cleaned = cleaned.str.replace(self.text_patterns['special_chars'], '', regex=True)
        
        # Fix multiple commas
        cleaned = cleaned.str.replace(self.text_patterns['multiple_commas'], ',', regex=True)
        
        # Strip leading/trailing spaces
        cleaned = cleaned.str.strip()
        
        # Field-specific cleaning
        if field_type == 'job_title':
            cleaned = self._clean_job_titles(cleaned)
        elif field_type == 'company':
            cleaned = self._clean_company_names(cleaned)
        elif field_type == 'location':
            cleaned = self._clean_locations(cleaned)
        elif field_type == 'skills':
            cleaned = self._clean_skills(cleaned)
        
        # Replace empty strings with NaN
        cleaned = cleaned.replace('', np.nan)
        
        return cleaned
    
    def _clean_job_titles(self, series):
        """Clean job titles specifically"""
        # Convert to title case
        cleaned = series.str.title()
        
        # Fix common abbreviations and terms
        replacements = {
            'Jr.': 'Junior',
            'Sr.': 'Senior', 
            'Mgr': 'Manager',
            'Mgmt': 'Management',
            'Assoc': 'Associate',
            'Asst': 'Assistant',
            'Dev': 'Developer',
            'Eng': 'Engineer',
            'Tech': 'Technical',
            'Ops': 'Operations',
            'Hr': 'HR',
            'It': 'IT',
            'Ui': 'UI',
            'Ux': 'UX',
            'Ai': 'AI',
            'Ml': 'ML'
        }
        
        for abbrev, full in replacements.items():
            cleaned = cleaned.str.replace(f'\\b{abbrev}\\b', full, regex=True)
        
        return cleaned
    
    def _clean_company_names(self, series):
        """Clean company names"""
        # Convert to title case
        cleaned = series.str.title()
        
        # Standardize company suffixes
        for standard, variations in self.company_patterns.items():
            for variation in variations:
                pattern = f'\\b{re.escape(variation)}\\b'
                cleaned = cleaned.str.replace(pattern, standard.upper(), regex=True, case=False)
        
        return cleaned
    
    def _clean_locations(self, series):
        """Clean and standardize location names"""
        # Convert to lowercase for mapping
        cleaned = series.str.lower().str.strip()
        
        # Apply location mappings
        for original, standard in self.location_mapping.items():
            cleaned = cleaned.str.replace(f'\\b{re.escape(original)}\\b', standard, regex=True)
        
        # Convert back to title case
        cleaned = cleaned.str.title()
        
        # Extract just city name if full address
        cleaned = cleaned.str.split(',').str[0]  # Take first part before comma
        
        return cleaned
    
    def _clean_skills(self, series):
        """Clean skills data"""
        # Convert to title case
        cleaned = series.str.title()
        
        # Common skill standardizations
        skill_replacements = {
            'Javascript': 'JavaScript',
            'Nodejs': 'Node.js',
            'Reactjs': 'React.js',
            'Vuejs': 'Vue.js',
            'Mysql': 'MySQL',
            'Postgresql': 'PostgreSQL',
            'Mongodb': 'MongoDB',
            'Aws': 'AWS',
            'Css3': 'CSS3',
            'Html5': 'HTML5',
            'C++': 'C++',
            'C#': 'C#',
            'Php': 'PHP'
        }
        
        for incorrect, correct in skill_replacements.items():
            cleaned = cleaned.str.replace(f'\\b{incorrect}\\b', correct, regex=True)
        
        return cleaned
    
    def handle_missing_values(self, df, strategy='smart'):
        """Handle missing values based on column type and context"""
        df_cleaned = df.copy()
        
        for column in df_cleaned.columns:
            missing_count = df_cleaned[column].isnull().sum()
            missing_pct = (missing_count / len(df_cleaned)) * 100
            
            if missing_count == 0:
                continue
                
            # Strategy based on column type and missing percentage  
            if missing_pct > 80:
                # Drop columns with >80% missing data
                print(f"   🗑️  Dropping column '{column}' ({missing_pct:.1f}% missing)")
                df_cleaned.drop(columns=[column], inplace=True)
                
            elif column in ['job_title', 'company', 'company_name']:
                # Drop rows missing critical job/company info
                df_cleaned = df_cleaned.dropna(subset=[column])
                print(f"   🗑️  Dropped {missing_count} rows missing '{column}'")
                
            elif 'salary' in column.lower() or 'income' in column.lower():
                # Fill salary fields with median or 0
                if df_cleaned[column].dtype in ['float64', 'int64']:
                    median_val = df_cleaned[column].median()
                    df_cleaned[column].fillna(median_val, inplace=True)
                    print(f"   💰 Filled '{column}' missing values with median: {median_val}")
                else:
                    df_cleaned[column].fillna('Not Disclosed', inplace=True)
                    
            elif column in ['location', 'city', 'state']:
                # Fill location with 'Unknown' 
                df_cleaned[column].fillna('Unknown', inplace=True)
                print(f"   📍 Filled '{column}' missing values with 'Unknown'")
                
            elif df_cleaned[column].dtype == 'object':
                # Fill text columns with 'Not Specified'
                df_cleaned[column].fillna('Not Specified', inplace=True)
                print(f"   📝 Filled '{column}' missing values with 'Not Specified'")
                
            elif df_cleaned[column].dtype in ['float64', 'int64']:
                # Fill numeric columns with median
                median_val = df_cleaned[column].median()
                df_cleaned[column].fillna(median_val, inplace=True)
                print(f"   🔢 Filled '{column}' missing values with median: {median_val}")
        
        return df_cleaned
    
    def remove_duplicates(self, df, subset_columns=None):
        """Remove duplicate rows"""
        initial_count = len(df)
        
        if subset_columns:
            # Remove duplicates based on specific columns
            df_dedup = df.drop_duplicates(subset=subset_columns, keep='first')
        else:
            # Remove complete duplicates
            df_dedup = df.drop_duplicates(keep='first')
        
        removed_count = initial_count - len(df_dedup)
        
        if removed_count > 0:
            print(f"   🗑️  Removed {removed_count} duplicate rows")
        
        return df_dedup
    
    def clean_single_dataset(self, filename):
        """Clean a single dataset"""
        print(f"\n🧹 Cleaning: {filename}")
        
        try:
            # Load standardized data
            df = pd.read_csv(self.processed_data_path / filename, low_memory=False)
            initial_shape = df.shape
            
            # Text field cleaning based on dataset type
            if 'job' in filename.lower() or 'posting' in filename.lower():
                self._clean_job_dataset(df)
            elif 'company' in filename.lower():
                self._clean_company_dataset(df)
            elif 'salary' in filename.lower():
                self._clean_salary_dataset(df)
            elif 'glassdoor' in filename.lower():
                self._clean_glassdoor_dataset(df)
            else:
                self._clean_general_dataset(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Remove duplicates
            df = self.remove_duplicates(df)
            
            # Save cleaned dataset
            output_filename = filename.replace('standardized_', 'cleaned_')
            output_path = self.cleaned_data_path / output_filename
            df.to_csv(output_path, index=False)
            
            final_shape = df.shape
            
            print(f"   ✅ Shape: {initial_shape} → {final_shape}")
            print(f"   ✅ Saved: {output_filename}")
            
            return {
                'success': True,
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'output_file': output_filename
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _clean_job_dataset(self, df):
        """Clean job-specific datasets"""
        if 'job_title' in df.columns:
            df['job_title'] = self.clean_text_field(df['job_title'], 'job_title')
        
        if 'company' in df.columns:
            df['company'] = self.clean_text_field(df['company'], 'company')
        elif 'company_name' in df.columns:
            df['company_name'] = self.clean_text_field(df['company_name'], 'company')
        
        if 'location' in df.columns:
            df['location'] = self.clean_text_field(df['location'], 'location')
        
        if 'job_description' in df.columns:
            df['job_description'] = self.clean_text_field(df['job_description'], 'general')
    
    def _clean_company_dataset(self, df):
        """Clean company-specific datasets"""
        if 'company_name' in df.columns:
            df['company_name'] = self.clean_text_field(df['company_name'], 'company')
        elif 'name' in df.columns:
            df['name'] = self.clean_text_field(df['name'], 'company')
        
        if 'city' in df.columns:
            df['city'] = self.clean_text_field(df['city'], 'location')
        
        if 'state' in df.columns:
            df['state'] = self.clean_text_field(df['state'], 'location')
    
    def _clean_salary_dataset(self, df):
        """Clean salary-specific datasets"""
        if 'job_title' in df.columns:
            df['job_title'] = self.clean_text_field(df['job_title'], 'job_title')
        
        if 'company' in df.columns:
            df['company'] = self.clean_text_field(df['company'], 'company')
        elif 'company_name' in df.columns:
            df['company_name'] = self.clean_text_field(df['company_name'], 'company')
    
    def _clean_glassdoor_dataset(self, df):
        """Clean Glassdoor-specific datasets"""
        self._clean_job_dataset(df)  # Use job dataset cleaning
    
    def _clean_general_dataset(self, df):
        """Clean general datasets"""
        # Clean any text columns
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            if col not in ['country_code', 'series_code']:  # Skip code columns
                df[col] = self.clean_text_field(df[col], 'general')
    
    def process_all_datasets(self):
        """Process all standardized datasets"""
        print("🧹 PHASE 5: DATA CLEANING")
        print("="*50)
        
        standardized_files = [f for f in os.listdir(self.processed_data_path) 
                            if f.startswith('standardized_') and f.endswith('.csv')]
        
        cleaning_report = {}
        
        for filename in standardized_files:
            result = self.clean_single_dataset(filename)
            cleaning_report[filename] = result
        
        # Save cleaning report
        report_path = self.cleaned_data_path / 'cleaning_report.json'
        with open(report_path, 'w') as f:
            json.dump(cleaning_report, f, indent=2, default=str)
        
        self.generate_summary(cleaning_report)
        return cleaning_report
    
    def generate_summary(self, report):
        """Generate cleaning summary"""
        print(f"\n📊 CLEANING SUMMARY")
        print("="*50)
        
        successful = len([k for k, v in report.items() if v.get('success', False)])
        failed = len([k for k, v in report.items() if not v.get('success', False)])
        
        print(f"✅ Successfully cleaned: {successful} datasets")
        print(f"❌ Failed to clean: {failed} datasets")
        
        # Calculate total data reduction
        total_initial_rows = sum(v.get('initial_shape', [0])[0] for v in report.values() if v.get('success'))
        total_final_rows = sum(v.get('final_shape', [0])[0] for v in report.values() if v.get('success'))
        reduction_pct = ((total_initial_rows - total_final_rows) / total_initial_rows * 100) if total_initial_rows > 0 else 0
        
        print(f"📊 Total rows: {total_initial_rows:,} → {total_final_rows:,} (-{reduction_pct:.1f}%)")
        
        print(f"\n🎯 KEY CLEANED DATASETS:")
        key_datasets = ['postings', 'companies_india', 'glassdoor_jobs', 'salary_data_cleaned']
        
        for dataset_key in key_datasets:
            for filename, info in report.items():
                if any(key in filename.lower() for key in dataset_key.split('_')):
                    if info.get('success'):
                        shape = info['final_shape']
                        print(f"   ✅ {filename:<35} {shape[0]:>6,} rows, {shape[1]:>2} cols")
                    break

def main():
    """Main execution function"""
    processed_data_path = Path(__file__).parent.parent / 'Processed_Data'
    cleaned_data_path = Path(__file__).parent.parent / 'Cleaned_Data'
    
    if not processed_data_path.exists():
        print("❌ Processed_Data directory not found!")
        print("➡️  Please run phase_04_column_standardization.py first")
        return
    
    cleaner = DataCleaner(processed_data_path, cleaned_data_path)
    report = cleaner.process_all_datasets()
    
    print("\n🎉 Phase 5 Complete: Data cleaning finished!")
    print("➡️  Next: Run phase_06_salary_normalization.py")

if __name__ == "__main__":
    main()