"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 7: EXPERIENCE EXTRACTION  
=================================

This script extracts experience requirements from job descriptions and
converts them into numeric min/max experience values.

Key Operations:
- Extract experience ranges from text (2-5 years, 3+ years, etc.)
- Handle various formats (fresher, entry level, senior, etc.)
- Create standardized experience_min, experience_max columns
- Map experience levels to numeric values

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

class ExperienceExtractor:
    def __init__(self, normalized_data_path, experience_data_path):
        self.normalized_data_path = Path(normalized_data_path)
        self.experience_data_path = Path(experience_data_path)
        self.experience_data_path.mkdir(exist_ok=True)
        
        # Experience extraction patterns
        self.experience_patterns = {
            # Numeric ranges
            'years_range': r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)',
            'years_plus': r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|y)',
            'years_to': r'(\d+(?:\.\d+)?)\s*(?:to|or)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)',
            'years_upto': r'(?:upto|up to|maximum)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)',
            'years_minimum': r'(?:minimum|min|atleast|at least)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)',
            
            # Experience levels
            'fresher': r'(?:fresher|fresh graduate|0 years?|no experience)',
            'entry_level': r'(?:entry level|junior|associate|trainee|intern)',
            'mid_level': r'(?:mid level|middle|experienced|senior associate)',
            'senior_level': r'(?:senior|lead|principal|manager|expert)',
            'executive': r'(?:director|vp|vice president|head|chief|cxo)',
            
            # Month-based patterns  
            'months_range': r'(\d+)\s*-\s*(\d+)\s*(?:months?|mons?)',
            'months_plus': r'(\d+)\s*\+?\s*(?:months?|mons?)',
        }
        
        # Experience level mappings (in years)
        self.experience_levels = {
            'fresher': (0, 0),
            'entry_level': (0, 2),
            'mid_level': (3, 7),
            'senior_level': (8, 15),
            'executive': (15, 30)
        }
        
    def extract_experience_from_text(self, text_field):
        """Extract experience requirements from text"""
        if pd.isna(text_field) or text_field == '':
            return None, None
            
        text = str(text_field).lower().strip()
        
        # Try numeric patterns first (most specific)
        for pattern_name, pattern in self.experience_patterns.items():
            if 'level' not in pattern_name and 'fresher' not in pattern_name:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return self._process_experience_match(match, pattern_name)
        
        # Try experience level patterns
        for level_name, (min_exp, max_exp) in self.experience_levels.items():
            if re.search(self.experience_patterns.get(level_name, ''), text, re.IGNORECASE):
                return min_exp, max_exp
        
        return None, None
    
    def _process_experience_match(self, match, pattern_name):
        """Process regex match and convert to years"""
        try:
            groups = match.groups()
            
            if 'range' in pattern_name or 'to' in pattern_name:
                if len(groups) >= 2:
                    min_exp = float(groups[0])
                    max_exp = float(groups[1])
                else:
                    min_exp = max_exp = float(groups[0])
            elif 'plus' in pattern_name or 'minimum' in pattern_name:
                min_exp = float(groups[0])
                max_exp = min_exp + 5  # Add 5 years for + requirements
            elif 'upto' in pattern_name:
                min_exp = 0
                max_exp = float(groups[0])
            else:
                min_exp = max_exp = float(groups[0])
            
            # Convert months to years if needed
            if 'months' in pattern_name:
                min_exp /= 12
                max_exp /= 12
            
            return min_exp, max_exp
            
        except (ValueError, IndexError):
            return None, None
    
    def extract_experience_level_from_title(self, job_title):
        """Extract experience level from job title"""
        if pd.isna(job_title):
            return None, None
            
        title = str(job_title).lower()
        
        # Senior positions
        if any(word in title for word in ['senior', 'sr', 'lead', 'principal', 'manager', 'head']):
            return 5, 12
        
        # Entry level positions  
        elif any(word in title for word in ['junior', 'jr', 'associate', 'trainee', 'intern', 'entry']):
            return 0, 3
        
        # Mid level (default for most positions)
        else:
            return 2, 8
    
    def normalize_experience_column(self, df, text_column, title_column=None):
        """Extract experience from text column and optionally job title"""
        if text_column not in df.columns:
            return df
        
        print(f"   📈 Extracting experience from: {text_column}")
        
        # Create new columns
        min_col = f"{text_column}_experience_min"
        max_col = f"{text_column}_experience_max"
        level_col = f"{text_column}_experience_level"
        
        df[min_col] = np.nan
        df[max_col] = np.nan
        df[level_col] = 'Not Specified'
        
        extracted_count = 0
        
        # Extract from text column
        for idx, text_value in df[text_column].items():
            min_exp, max_exp = self.extract_experience_from_text(text_value)
            
            if min_exp is not None and max_exp is not None:
                df.at[idx, min_col] = min_exp
                df.at[idx, max_col] = max_exp
                df.at[idx, level_col] = self._categorize_experience_level(min_exp, max_exp)
                extracted_count += 1
        
        # Fill missing values from job title if available
        if title_column and title_column in df.columns:
            missing_mask = df[min_col].isna()
            missing_count = missing_mask.sum()
            
            if missing_count > 0:
                print(f"   🔍 Extracting from job titles for {missing_count} missing entries")
                
                for idx in df[missing_mask].index:
                    job_title = df.at[idx, title_column]
                    min_exp, max_exp = self.extract_experience_level_from_title(job_title)
                    
                    if min_exp is not None:
                        df.at[idx, min_col] = min_exp
                        df.at[idx, max_col] = max_exp
                        df.at[idx, level_col] = self._categorize_experience_level(min_exp, max_exp)
                        extracted_count += 1
        
        print(f"   ✅ Extracted experience for {extracted_count}/{len(df)} entries")
        
        return df
    
    def _categorize_experience_level(self, min_exp, max_exp):
        """Categorize experience into levels"""
        avg_exp = (min_exp + max_exp) / 2
        
        if avg_exp == 0:
            return 'Fresher'
        elif avg_exp <= 2:
            return 'Entry Level'
        elif avg_exp <= 5:
            return 'Junior Level'
        elif avg_exp <= 10:
            return 'Mid Level'
        elif avg_exp <= 15:
            return 'Senior Level'
        else:
            return 'Executive Level'
    
    def process_single_dataset(self, filename):
        """Process experience extraction for single dataset"""
        print(f"\n📈 Processing experience: {filename}")
        
        try:
            # Load normalized data
            df = pd.read_csv(self.normalized_data_path / filename, low_memory=False)
            initial_shape = df.shape
            
            # Identify text columns that might contain experience info
            experience_columns = []
            title_column = None
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['description', 'requirement', 'qualification', 'experience']):
                    experience_columns.append(col)
                elif 'title' in col_lower or col_lower == 'job' or col_lower == 'role':
                    title_column = col
            
            if not experience_columns and not title_column:
                print(f"   ℹ️  No experience-related columns found")
                # Copy file as-is
                output_filename = filename.replace('normalized_', 'experience_')
                output_path = self.experience_data_path / output_filename
                df.to_csv(output_path, index=False)
                
                return {
                    'success': True,
                    'experience_columns': [],
                    'initial_shape': initial_shape,
                    'final_shape': df.shape,
                    'output_file': output_filename
                }
            
            print(f"   📊 Found columns: {experience_columns + ([title_column] if title_column else [])}")
            
            # Process each experience column
            for col in experience_columns:
                df = self.normalize_experience_column(df, col, title_column)
            
            # If no experience columns but have title, extract from title
            if not experience_columns and title_column:
                df = self.normalize_experience_column(df, title_column)
            
            # Add overall experience summary columns
            df = self._create_experience_summary(df)
            
            # Save processed dataset
            output_filename = filename.replace('normalized_', 'experience_')
            output_path = self.experience_data_path / output_filename
            df.to_csv(output_path, index=False)
            
            final_shape = df.shape
            
            print(f"   ✅ Shape: {initial_shape} → {final_shape}")
            print(f"   ✅ Saved: {output_filename}")
            
            return {
                'success': True,
                'experience_columns': experience_columns,
                'title_column': title_column,
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'output_file': output_filename
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_experience_summary(self, df):
        """Create summary experience columns from all extracted experience data"""
        # Find all experience min/max columns
        exp_min_cols = [col for col in df.columns if 'experience_min' in col]
        exp_max_cols = [col for col in df.columns if 'experience_max' in col]
        
        if exp_min_cols and exp_max_cols:
            # Create overall experience requirements
            df['required_experience_min'] = df[exp_min_cols].min(axis=1, skipna=True)
            df['required_experience_max'] = df[exp_max_cols].max(axis=1, skipna=True)
            
            # Create experience category
            df['experience_category'] = df.apply(
                lambda row: self._categorize_experience_level(
                    row.get('required_experience_min', 0),
                    row.get('required_experience_max', 0)
                ) if pd.notna(row.get('required_experience_min')) else 'Not Specified',
                axis=1
            )
            
            print(f"   📊 Created experience summary columns")
        
        return df
    
    def process_all_datasets(self):
        """Process all normalized datasets"""
        print("📈 PHASE 7: EXPERIENCE EXTRACTION")
        print("="*50)
        
        normalized_files = [f for f in os.listdir(self.normalized_data_path) 
                           if f.startswith('normalized_') and f.endswith('.csv')]
        
        experience_report = {}
        
        for filename in normalized_files:
            result = self.process_single_dataset(filename)
            experience_report[filename] = result
        
        # Save experience report
        report_path = self.experience_data_path / 'experience_extraction_report.json'
        with open(report_path, 'w') as f:
            json.dump(experience_report, f, indent=2, default=str)
        
        self.generate_summary(experience_report)
        return experience_report
    
    def generate_summary(self, report):
        """Generate experience extraction summary"""
        print(f"\n📊 EXPERIENCE EXTRACTION SUMMARY")
        print("="*50)
        
        successful = len([k for k, v in report.items() if v.get('success', False)])
        failed = len([k for k, v in report.items() if not v.get('success', False)])
        
        print(f"✅ Successfully processed: {successful} datasets")
        print(f"❌ Failed to process: {failed} datasets")
        
        # Count datasets with experience columns
        with_experience = len([k for k, v in report.items() 
                              if v.get('success') and (len(v.get('experience_columns', [])) > 0 or v.get('title_column'))])
        
        print(f"📈 Datasets with experience data: {with_experience}")
        
        print(f"\n🎯 KEY EXPERIENCE DATASETS:")
        job_datasets = ['postings', 'glassdoor_jobs', 'salary_data_cleaned', 'eda_data']
        
        for dataset_key in job_datasets:
            for filename, info in report.items():
                if any(key in filename.lower() for key in dataset_key.split('_')):
                    if info.get('success'):
                        exp_cols = len(info.get('experience_columns', []))
                        title_col = 1 if info.get('title_column') else 0
                        shape = info['final_shape']
                        print(f"   📈 {filename:<35} {exp_cols + title_col:>2} text cols, {shape[0]:>6,} rows")
                    break

def main():
    """Main execution function"""
    normalized_data_path = Path(__file__).parent.parent / 'Normalized_Data'
    experience_data_path = Path(__file__).parent.parent / 'Experience_Data'
    
    if not normalized_data_path.exists():
        print("❌ Normalized_Data directory not found!")
        print("➡️  Please run phase_06_salary_normalization.py first")
        return
    
    extractor = ExperienceExtractor(normalized_data_path, experience_data_path)
    report = extractor.process_all_datasets()
    
    print("\n🎉 Phase 7 Complete: Experience extraction finished!")
    print("➡️  Ready for Phase 8-13 (will be provided next)")

if __name__ == "__main__":
    main()