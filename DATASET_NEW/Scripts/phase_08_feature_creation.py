"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 8: FEATURE CREATION
=================================

This script creates new useful features for MaaSarthi ML models including:
- Remote job detection
- Source file tracking
- Job category classification
- Seniority level extraction
- India-specific features

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

class FeatureCreator:
    def __init__(self, experience_data_path, feature_data_path):
        self.experience_data_path = Path(experience_data_path)
        self.feature_data_path = Path(feature_data_path)
        self.feature_data_path.mkdir(exist_ok=True)
        
        # Remote work keywords
        self.remote_keywords = [
            'remote', 'work from home', 'wfh', 'telecommute', 'virtual', 
            'distributed', 'anywhere', 'home-based', 'online', 'digital nomad'
        ]
        
        # Job category mappings
        self.job_categories = {
            'Technology': [
                'software', 'developer', 'engineer', 'programmer', 'tech', 'it', 
                'data scientist', 'analyst', 'web', 'mobile', 'devops', 'cloud'
            ],
            'Marketing': [
                'marketing', 'digital marketing', 'seo', 'content', 'social media', 
                'brand', 'advertising', 'campaign', 'growth'
            ],
            'Sales': [
                'sales', 'business development', 'account manager', 'customer', 
                'client', 'revenue', 'lead generation'
            ],
            'Design': [
                'designer', 'ui', 'ux', 'graphic', 'creative', 'visual', 'art', 
                'illustration', 'branding'
            ],
            'Finance': [
                'finance', 'accounting', 'financial', 'analyst', 'investment', 
                'banking', 'audit', 'tax', 'payroll'
            ],
            'Human Resources': [
                'hr', 'human resources', 'recruiter', 'talent', 'people', 
                'organizational', 'training', 'employee'
            ],
            'Operations': [
                'operations', 'logistics', 'supply chain', 'project manager', 
                'coordinator', 'administration', 'office manager'
            ],
            'Healthcare': [
                'doctor', 'nurse', 'medical', 'healthcare', 'physician', 
                'therapist', 'pharmacist', 'clinical'
            ],
            'Education': [
                'teacher', 'tutor', 'education', 'training', 'instructor', 
                'academic', 'curriculum', 'learning'
            ],
            'Customer Service': [
                'customer service', 'support', 'help desk', 'call center', 
                'customer care', 'service representative'
            ]
        }
        
        # Indian cities for location-based features
        self.indian_metros = ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata']
        self.indian_tier1 = ['pune', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur', 'nagpur']
        self.indian_tier2 = ['coimbatore', 'kochi', 'indore', 'bhopal', 'visakhapatnam', 'vadodara']
        
    def detect_remote_jobs(self, df):
        """Detect if job allows remote work"""
        print("   🏠 Detecting remote work opportunities...")
        
        # Initialize remote flags
        df['remote_flag'] = False
        df['work_arrangement'] = 'On-site'
        
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['company_id', 'job_id']:
                text_columns.append(col)
        
        remote_count = 0
        
        for idx, row in df.iterrows():
            # Check all text columns for remote keywords
            is_remote = False
            text_content = ""
            
            for col in text_columns:
                if pd.notna(row[col]):
                    text_content += str(row[col]).lower() + " "
            
            # Check for remote work indicators
            for keyword in self.remote_keywords:
                if keyword in text_content:
                    is_remote = True
                    break
            
            if is_remote:
                df.at[idx, 'remote_flag'] = True
                df.at[idx, 'work_arrangement'] = 'Remote'
                remote_count += 1
            elif any(keyword in text_content for keyword in ['hybrid', 'flexible']):
                df.at[idx, 'work_arrangement'] = 'Hybrid'
        
        print(f"   ✅ Found {remote_count} remote jobs ({remote_count/len(df)*100:.1f}%)")
        return df
    
    def add_source_tracking(self, df, filename):
        """Add source file tracking"""
        print("   📂 Adding source tracking...")
        
        # Extract source from filename
        if 'postings' in filename.lower():
            source = 'Job Postings'
        elif 'glassdoor' in filename.lower():
            source = 'Glassdoor'
        elif 'salary' in filename.lower():
            source = 'Salary Data'
        elif 'company' in filename.lower():
            source = 'Company Data'
        elif 'eda' in filename.lower():
            source = 'EDA Dataset'
        else:
            source = 'Other'
        
        df['source_file'] = filename
        df['data_source'] = source
        df['processing_date'] = '2026-03-04'
        
        print(f"   ✅ Source: {source}")
        return df
    
    def classify_job_categories(self, df):
        """Classify jobs into categories"""
        print("   🏷️  Classifying job categories...")
        
        df['job_category'] = 'Other'
        df['job_subcategory'] = 'General'
        
        # Look for job title column
        title_col = None
        for col in ['job_title', 'title', 'position', 'role']:
            if col in df.columns:
                title_col = col
                break
        
        if title_col is None:
            print("   ⚠️  No job title column found, skipping categorization")
            return df
        
        categorized_count = 0
        
        for idx, title in df[title_col].items():
            if pd.notna(title):
                title_lower = str(title).lower()
                
                for category, keywords in self.job_categories.items():
                    if any(keyword in title_lower for keyword in keywords):
                        df.at[idx, 'job_category'] = category
                        # Find most specific subcategory
                        for keyword in keywords:
                            if keyword in title_lower:
                                df.at[idx, 'job_subcategory'] = keyword.title()
                                break
                        categorized_count += 1
                        break
        
        print(f"   ✅ Categorized {categorized_count}/{len(df)} jobs")
        return df
    
    def extract_seniority_level(self, df):
        """Extract seniority level from job titles"""
        print("   📈 Extracting seniority levels...")
        
        df['seniority_level'] = 'Mid-Level'
        df['leadership_role'] = False
        
        # Find job title column
        title_col = None
        for col in ['job_title', 'title', 'position', 'role']:
            if col in df.columns:
                title_col = col
                break
        
        if title_col is None:
            return df
        
        seniority_patterns = {
            'Entry-Level': ['intern', 'trainee', 'junior', 'associate', 'entry', 'graduate', 'fresher'],
            'Mid-Level': ['analyst', 'specialist', 'coordinator', 'executive'],
            'Senior-Level': ['senior', 'lead', 'principal', 'expert', 'consultant'],
            'Management': ['manager', 'supervisor', 'team lead', 'head'],
            'Executive': ['director', 'vp', 'vice president', 'ceo', 'cto', 'cfo', 'chief']
        }
        
        for idx, title in df[title_col].items():
            if pd.notna(title):
                title_lower = str(title).lower()
                
                for level, keywords in seniority_patterns.items():
                    if any(keyword in title_lower for keyword in keywords):
                        df.at[idx, 'seniority_level'] = level
                        
                        # Mark leadership roles
                        if level in ['Management', 'Executive']:
                            df.at[idx, 'leadership_role'] = True
                        break
        
        return df
    
    def add_location_features(self, df):
        """Add India-specific location features"""
        print("   📍 Adding location features...")
        
        df['city_tier'] = 'Unknown'
        df['is_metro'] = False
        df['is_india'] = False
        
        # Find location columns
        location_cols = []
        for col in ['location', 'city', 'state', 'address']:
            if col in df.columns:
                location_cols.append(col)
        
        if not location_cols:
            return df
        
        for idx, row in df.iterrows():
            location_text = ""
            for col in location_cols:
                if pd.notna(row[col]):
                    location_text += str(row[col]).lower() + " "
            
            # Check if it's in India
            if any(city in location_text for city in self.indian_metros + self.indian_tier1 + self.indian_tier2):
                df.at[idx, 'is_india'] = True
                
                # Classify city tier
                if any(city in location_text for city in self.indian_metros):
                    df.at[idx, 'city_tier'] = 'Metro'
                    df.at[idx, 'is_metro'] = True
                elif any(city in location_text for city in self.indian_tier1):
                    df.at[idx, 'city_tier'] = 'Tier-1'
                elif any(city in location_text for city in self.indian_tier2):
                    df.at[idx, 'city_tier'] = 'Tier-2'
                else:
                    df.at[idx, 'city_tier'] = 'Tier-3'
            elif 'india' in location_text or 'indian' in location_text:
                df.at[idx, 'is_india'] = True
                df.at[idx, 'city_tier'] = 'Other India'
        
        india_count = df['is_india'].sum()
        print(f"   ✅ Found {india_count} India-based positions ({india_count/len(df)*100:.1f}%)")
        return df
    
    def add_salary_features(self, df):
        """Add salary-based features"""
        print("   💰 Creating salary features...")
        
        # Find salary columns
        salary_cols = [col for col in df.columns if 'salary' in col.lower() and ('min' in col or 'max' in col or 'avg' in col)]
        
        if not salary_cols:
            return df
        
        # Create salary affordability categories (for Indian market)
        avg_salary_col = None
        for col in salary_cols:
            if 'avg' in col.lower():
                avg_salary_col = col
                break
        
        if avg_salary_col:
            df['salary_bracket'] = df[avg_salary_col].apply(self._categorize_indian_salary)
            df['is_competitive_salary'] = df[avg_salary_col] > df[avg_salary_col].median()
        
        return df
    
    def _categorize_indian_salary(self, salary):
        """Categorize salary for Indian market"""
        if pd.isna(salary):
            return 'Not Disclosed'
        
        # Convert to lakhs for easier understanding
        salary_lpa = salary / 100000
        
        if salary_lpa < 2:
            return 'Below 2 LPA'
        elif salary_lpa < 5:
            return '2-5 LPA'
        elif salary_lpa < 10:
            return '5-10 LPA'
        elif salary_lpa < 20:
            return '10-20 LPA'
        elif salary_lpa < 50:
            return '20-50 LPA'
        else:
            return '50+ LPA'
    
    def add_maasarthi_specific_features(self, df):
        """Add MaaSarthi-specific features for women's employment"""
        print("   👩‍💼 Adding MaaSarthi-specific features...")
        
        # Female-friendly job indicators
        df['female_friendly'] = False
        df['flexible_hours'] = False
        df['part_time_available'] = False
        
        # Look for female-friendly keywords in job descriptions
        female_friendly_keywords = [
            'flexible', 'work life balance', 'maternity', 'diversity', 'inclusive', 
            'equal opportunity', 'women', 'female', 'part time', 'flexible hours'
        ]
        
        flexible_keywords = [
            'flexible schedule', 'flexible timing', 'work from home', 'remote',
            'part time', 'freelance', 'contract', 'flexible hours'
        ]
        
        # Check text columns for these keywords
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        
        for idx, row in df.iterrows():
            text_content = ""
            for col in text_columns:
                if pd.notna(row[col]):
                    text_content += str(row[col]).lower() + " "
            
            # Check for female-friendly indicators
            if any(keyword in text_content for keyword in female_friendly_keywords):
                df.at[idx, 'female_friendly'] = True
            
            # Check for flexible work arrangements
            if any(keyword in text_content for keyword in flexible_keywords):
                df.at[idx, 'flexible_hours'] = True
            
            # Check for part-time availability
            if any(keyword in text_content for keyword in ['part time', 'part-time', 'freelance', 'contract']):
                df.at[idx, 'part_time_available'] = True
        
        # Job suitability score for working mothers
        df['mother_suitability_score'] = 0
        
        for idx, row in df.iterrows():
            score = 0
            
            # Add points for remote work
            if row.get('remote_flag', False):
                score += 3
            
            # Add points for flexible arrangements
            if row.get('flexible_hours', False):
                score += 2
            
            # Add points for part-time options
            if row.get('part_time_available', False):
                score += 2
            
            # Add points for female-friendly companies
            if row.get('female_friendly', False):
                score += 1
            
            # Add points for certain job categories
            if row.get('job_category') in ['Education', 'Healthcare', 'Customer Service']:
                score += 1
            
            df.at[idx, 'mother_suitability_score'] = min(score, 10)  # Cap at 10
        
        return df
    
    def process_single_dataset(self, filename):
        """Add features to single dataset"""
        print(f"\n🔧 Adding features: {filename}")
        
        try:
            # Load experience data
            df = pd.read_csv(self.experience_data_path / filename, low_memory=False)
            initial_shape = df.shape
            
            # Add all features
            df = self.add_source_tracking(df, filename)
            df = self.detect_remote_jobs(df)
            df = self.classify_job_categories(df)
            df = self.extract_seniority_level(df)
            df = self.add_location_features(df)
            df = self.add_salary_features(df)
            df = self.add_maasarthi_specific_features(df)
            
            # Save enhanced dataset
            output_filename = filename.replace('experience_', 'featured_')
            output_path = self.feature_data_path / output_filename
            df.to_csv(output_path, index=False)
            
            final_shape = df.shape
            new_features = final_shape[1] - initial_shape[1]
            
            print(f"   ✅ Shape: {initial_shape} → {final_shape} (+{new_features} features)")
            print(f"   ✅ Saved: {output_filename}")
            
            return {
                'success': True,
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'new_features': new_features,
                'output_file': output_filename
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_all_datasets(self):
        """Process all experience datasets"""
        print("🔧 PHASE 8: FEATURE CREATION")
        print("="*50)
        
        experience_files = [f for f in os.listdir(self.experience_data_path) 
                           if f.startswith('experience_') and f.endswith('.csv')]
        
        feature_report = {}
        
        for filename in experience_files:
            result = self.process_single_dataset(filename)
            feature_report[filename] = result
        
        # Save feature report
        report_path = self.feature_data_path / 'feature_creation_report.json'
        with open(report_path, 'w') as f:
            json.dump(feature_report, f, indent=2, default=str)
        
        self.generate_summary(feature_report)
        return feature_report
    
    def generate_summary(self, report):
        """Generate feature creation summary"""
        print(f"\n📊 FEATURE CREATION SUMMARY")
        print("="*50)
        
        successful = len([k for k, v in report.items() if v.get('success', False)])
        failed = len([k for k, v in report.items() if not v.get('success', False)])
        
        print(f"✅ Successfully processed: {successful} datasets")
        print(f"❌ Failed to process: {failed} datasets")
        
        total_features = sum(v.get('new_features', 0) for v in report.values() if v.get('success'))
        print(f"🔧 Total new features added: {total_features}")
        
        # Show feature-rich datasets
        print(f"\n🎯 KEY FEATURE-ENHANCED DATASETS:")
        key_datasets = ['postings', 'glassdoor_jobs', 'salary_data_cleaned', 'eda_data']
        
        for dataset_key in key_datasets:
            for filename, info in report.items():
                if any(key in filename.lower() for key in dataset_key.split('_')):
                    if info.get('success'):
                        shape = info['final_shape']
                        features = info.get('new_features', 0)
                        print(f"   🔧 {filename:<35} +{features:>2} features, {shape[0]:>6,} rows")
                    break

def main():
    """Main execution function"""
    experience_data_path = Path(__file__).parent.parent / 'Experience_Data'
    feature_data_path = Path(__file__).parent.parent / 'Feature_Data'
    
    if not experience_data_path.exists():
        print("❌ Experience_Data directory not found!")
        print("➡️  Please run phase_07_experience_extraction.py first")
        return
    
    creator = FeatureCreator(experience_data_path, feature_data_path)
    report = creator.process_all_datasets()
    
    print("\n🎉 Phase 8 Complete: Feature creation finished!")
    print("➡️  Next: Run phase_09_dataset_merging.py")

if __name__ == "__main__":
    main()