"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 6: SALARY NORMALIZATION
=================================

This script converts messy salary text into clean numeric values.
Converts various salary formats into standardized min/max salary columns.

Key Operations:
- Extract salary ranges from text (₹3-5 LPA, $50K-70K, etc.)
- Convert currencies (USD to INR, LPA to annual, etc.)
- Handle "Not Disclosed" and missing salary data
- Create standardized salary_min, salary_max columns

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

class SalaryNormalizer:
    def __init__(self, cleaned_data_path, normalized_data_path):
        self.cleaned_data_path = Path(cleaned_data_path)
        self.normalized_data_path = Path(normalized_data_path)
        self.normalized_data_path.mkdir(exist_ok=True)
        
        # Currency conversion rates (as of March 2026)
        self.currency_rates = {
            'USD': 83.0,   # 1 USD = 83 INR
            'EUR': 90.0,   # 1 EUR = 90 INR  
            'GBP': 105.0,  # 1 GBP = 105 INR
            'INR': 1.0,    # 1 INR = 1 INR
            'CAD': 62.0,   # 1 CAD = 62 INR
            'AUD': 55.0    # 1 AUD = 55 INR
        }
        
        # Salary patterns for extraction
        self.salary_patterns = {
            # Indian salary patterns
            'lpa_range': r'₹?(\d+(?:\.\d+)?)\s*-\s*₹?(\d+(?:\.\d+)?)\s*(?:lacs?|lpa|lakhs?)',
            'lpa_single': r'₹?(\d+(?:\.\d+)?)\s*(?:lacs?|lpa|lakhs?)',
            'inr_range': r'₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*-\s*₹?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            'inr_single': r'₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            
            # International salary patterns  
            'usd_k_range': r'\$(\d+(?:\.\d+)?)\s*k?\s*-\s*\$?(\d+(?:\.\d+)?)\s*k',
            'usd_k_single': r'\$(\d+(?:\.\d+)?)\s*k',
            'usd_range': r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*-\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            'usd_single': r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            
            # Per hour patterns
            'hourly_range': r'\$?(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)\s*/?\s*(?:hr|hour)',
            'hourly_single': r'\$?(\d+(?:\.\d+)?)\s*/?\s*(?:hr|hour)',
            
            # General number ranges
            'number_range': r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)',
            'number_single': r'(\d+(?:\.\d+)?)'
        }
        
        # Keywords indicating undisclosed salary
        self.undisclosed_keywords = [
            'not disclosed', 'confidential', 'competitive', 'negotiable', 
            'based on experience', 'market rate', 'as per company norms',
            'undisclosed', 'not specified', 'tbd', 'to be decided'
        ]
    
    def extract_salary_from_text(self, salary_text, currency='INR'):
        """Extract numeric salary values from text"""
        if pd.isna(salary_text) or salary_text == '':
            return None, None
            
        salary_text = str(salary_text).lower().strip()
        
        # Check for undisclosed salary keywords
        if any(keyword in salary_text for keyword in self.undisclosed_keywords):
            return None, None
        
        # Try different patterns in order of specificity
        for pattern_name, pattern in self.salary_patterns.items():
            match = re.search(pattern, salary_text, re.IGNORECASE)
            if match:
                return self._process_salary_match(match, pattern_name, currency)
        
        return None, None
    
    def _process_salary_match(self, match, pattern_name, currency):
        """Process regex match and convert to standardized format"""
        try:
            groups = match.groups()
            
            if 'range' in pattern_name and len(groups) >= 2:
                min_val = float(groups[0].replace(',', ''))
                max_val = float(groups[1].replace(',', ''))
            else:
                min_val = max_val = float(groups[0].replace(',', ''))
            
            # Apply multipliers and conversions based on pattern type
            if 'lpa' in pattern_name:
                # LPA (Lakhs Per Annum) - multiply by 100,000
                min_val *= 100000
                max_val *= 100000
            elif 'k' in pattern_name:
                # K format (e.g., 50K) - multiply by 1000
                min_val *= 1000
                max_val *= 1000
            elif 'hourly' in pattern_name:
                # Hourly rate - convert to annual (assuming 40 hours/week, 52 weeks/year)
                min_val *= 40 * 52
                max_val *= 40 * 52
            
            # Convert currency to INR
            if currency != 'INR' and currency in self.currency_rates:
                conversion_rate = self.currency_rates[currency]
                min_val *= conversion_rate
                max_val *= conversion_rate
            
            return min_val, max_val
            
        except (ValueError, IndexError) as e:
            return None, None
    
    def detect_currency(self, salary_text):
        """Detect currency from salary text"""
        if pd.isna(salary_text):
            return 'INR'
            
        salary_text = str(salary_text).lower()
        
        currency_indicators = {
            'INR': ['₹', 'inr', 'rupees', 'lacs', 'lakhs', 'lpa', 'crores'],
            'USD': ['$', 'usd', 'dollars', 'dollar'],
            'EUR': ['€', 'eur', 'euros', 'euro'],
            'GBP': ['£', 'gbp', 'pounds', 'pound'],
            'CAD': ['cad', 'canadian'],
            'AUD': ['aud', 'australian']
        }
        
        for currency, indicators in currency_indicators.items():
            if any(indicator in salary_text for indicator in indicators):
                return currency
        
        return 'INR'  # Default to INR for Indian dataset
    
    def normalize_salary_column(self, df, salary_column):
        """Normalize salary column in a dataframe"""
        if salary_column not in df.columns:
            return df
        
        print(f"   💰 Normalizing column: {salary_column}")
        
        # Create new columns for min and max salary
        min_col = f"{salary_column}_min"
        max_col = f"{salary_column}_max"
        currency_col = f"{salary_column}_currency"
        
        # Initialize new columns
        df[min_col] = np.nan
        df[max_col] = np.nan
        df[currency_col] = 'INR'
        
        processed_count = 0
        
        for idx, salary_text in df[salary_column].items():
            if pd.notna(salary_text) and str(salary_text).strip() != '':
                currency = self.detect_currency(salary_text)
                min_sal, max_sal = self.extract_salary_from_text(salary_text, currency)
                
                if min_sal is not None and max_sal is not None:
                    df.at[idx, min_col] = min_sal
                    df.at[idx, max_col] = max_sal
                    df.at[idx, currency_col] = currency
                    processed_count += 1
        
        print(f"   ✅ Processed {processed_count}/{len(df)} salary entries")
        
        return df
    
    def normalize_single_dataset(self, filename):
        """Normalize salariess in a single dataset"""
        print(f"\n💰 Normalizing salaries: {filename}")
        
        try:
            # Load cleaned data
            df = pd.read_csv(self.cleaned_data_path / filename, low_memory=False)
            initial_shape = df.shape
            
            # Identify salary columns
            salary_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['salary', 'compensation', 'pay', 'wage', 'income']):
                    salary_columns.append(col)
            
            if not salary_columns:
                print(f"   ℹ️  No salary columns found")
                # Just copy the file as-is
                output_filename = filename.replace('cleaned_', 'normalized_')
                output_path = self.normalized_data_path / output_filename
                df.to_csv(output_path, index=False)
                
                return {
                    'success': True,
                    'salary_columns': [],
                    'initial_shape': initial_shape,
                    'final_shape': df.shape,
                    'output_file': output_filename
                }
            
            print(f"   📊 Found salary columns: {salary_columns}")
            
            # Normalize each salary column
            for col in salary_columns:
                df = self.normalize_salary_column(df, col)
            
            # Additional processing for specific datasets
            if 'glassdoor' in filename.lower() or 'salary' in filename.lower():
                df = self._postprocess_salary_dataset(df)
            
            # Save normalized dataset
            output_filename = filename.replace('cleaned_', 'normalized_')
            output_path = self.normalized_data_path / output_filename
            df.to_csv(output_path, index=False)
            
            final_shape = df.shape
            
            print(f"   ✅ Shape: {initial_shape} → {final_shape}")
            print(f"   ✅ Saved: {output_filename}")
            
            return {
                'success': True,
                'salary_columns': salary_columns,
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'output_file': output_filename
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _postprocess_salary_dataset(self, df):
        """Additional processing for salary-specific datasets"""
        # Calculate average salary if both min and max exist
        salary_min_cols = [col for col in df.columns if col.endswith('_min')]
        salary_max_cols = [col for col in df.columns if col.endswith('_max')]
        
        for min_col in salary_min_cols:
            max_col = min_col.replace('_min', '_max')
            avg_col = min_col.replace('_min', '_avg')
            
            if max_col in df.columns:
                df[avg_col] = (df[min_col] + df[max_col]) / 2
                print(f"   📊 Created average column: {avg_col}")
        
        # Add salary categories for easier analysis  
        if any('salary' in col for col in df.columns):
            df['salary_category'] = df.apply(self._categorize_salary, axis=1)
            print(f"   🏷️  Added salary categories")
        
        return df
    
    def _categorize_salary(self, row):
        """Categorize salary into ranges for easier analysis"""
        # Find any salary_avg column
        avg_cols = [col for col in row.index if 'salary' in col.lower() and 'avg' in col.lower()]
        if not avg_cols:
            # Try to find min or max columns
            min_cols = [col for col in row.index if 'salary' in col.lower() and 'min' in col.lower()]
            if min_cols:
                avg_salary = row[min_cols[0]]
            else:
                return 'Unknown'
        else:
            avg_salary = row[avg_cols[0]]
        
        if pd.isna(avg_salary):
            return 'Not Disclosed'
        
        # Indian salary categories (in INR per annum)
        if avg_salary < 300000:  # < 3 LPA
            return 'Entry Level'
        elif avg_salary < 600000:  # 3-6 LPA
            return 'Junior Level'
        elif avg_salary < 1200000:  # 6-12 LPA
            return 'Mid Level'
        elif avg_salary < 2500000:  # 12-25 LPA
            return 'Senior Level'
        else:  # > 25 LPA
            return 'Executive Level'
    
    def process_all_datasets(self):
        """Process all cleaned datasets"""
        print("💰 PHASE 6: SALARY NORMALIZATION")
        print("="*50)
        
        cleaned_files = [f for f in os.listdir(self.cleaned_data_path) 
                        if f.startswith('cleaned_') and f.endswith('.csv')]
        
        normalization_report = {}
        
        for filename in cleaned_files:
            result = self.normalize_single_dataset(filename)
            normalization_report[filename] = result
        
        # Save normalization report
        report_path = self.normalized_data_path / 'salary_normalization_report.json'
        with open(report_path, 'w') as f:
            json.dump(normalization_report, f, indent=2, default=str)
        
        self.generate_summary(normalization_report)
        return normalization_report
    
    def generate_summary(self, report):
        """Generate normalization summary"""
        print(f"\n📊 SALARY NORMALIZATION SUMMARY")
        print("="*50)
        
        successful = len([k for k, v in report.items() if v.get('success', False)])
        failed = len([k for k, v in report.items() if not v.get('success', False)])
        
        print(f"✅ Successfully processed: {successful} datasets")
        print(f"❌ Failed to process: {failed} datasets")
        
        # Count datasets with salary columns
        with_salary = len([k for k, v in report.items() 
                          if v.get('success') and len(v.get('salary_columns', [])) > 0])
        
        print(f"💰 Datasets with salary data: {with_salary}")
        
        print(f"\n🎯 KEY SALARY DATASETS:")
        salary_datasets = ['postings', 'glassdoor_jobs', 'salary_data_cleaned', 'salaries']
        
        for dataset_key in salary_datasets:
            for filename, info in report.items():
                if any(key in filename.lower() for key in dataset_key.split('_')):
                    if info.get('success'):
                        salary_cols = info.get('salary_columns', [])
                        shape = info['final_shape']
                        print(f"   💰 {filename:<35} {len(salary_cols):>2} salary cols, {shape[0]:>6,} rows")
                    break

def main():
    """Main execution function"""
    cleaned_data_path = Path(__file__).parent.parent / 'Cleaned_Data'
    normalized_data_path = Path(__file__).parent.parent / 'Normalized_Data'
    
    if not cleaned_data_path.exists():
        print("❌ Cleaned_Data directory not found!")
        print("➡️  Please run phase_05_data_cleaning.py first")
        return
    
    normalizer = SalaryNormalizer(cleaned_data_path, normalized_data_path)
    report = normalizer.process_all_datasets()
    
    print("\n🎉 Phase 6 Complete: Salary normalization finished!")
    print("➡️  Next: Run phase_07_experience_extraction.py")

if __name__ == "__main__":
    main()