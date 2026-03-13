"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 10: DUPLICATE REMOVAL
=================================

This script identifies and removes duplicate job postings using:
- Job title similarity
- Company name matching  
- Location comparison
- Advanced duplicate detection algorithms

Author: MaaSarthi Data Team
Date: March 2026
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

class DuplicateRemover:
    def __init__(self, merged_data_path, deduplicated_data_path):
        self.merged_data_path = Path(merged_data_path)
        self.deduplicated_data_path = Path(deduplicated_data_path)
        self.deduplicated_data_path.mkdir(exist_ok=True)
        
        # Similarity thresholds for duplicate detection
        self.similarity_thresholds = {
            'job_title': 0.8,      # 80% similarity for job titles
            'company': 0.9,        # 90% similarity for company names
            'location': 0.7        # 70% similarity for locations
        }
        
        # Key columns for duplicate detection
        self.duplicate_check_columns = [
            'job_title', 'company', 'company_name', 'location'
        ]
    
    def normalize_text_for_comparison(self, text):
        """Normalize text for better duplicate detection"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        
        # Remove common variations
        replacements = {
            'pvt ltd': '',
            'private limited': '',
            'ltd': '',
            'inc': '',
            'corporation': '',
            'corp': '',
            'company': '',
            'co': '',
            '&': 'and',
            '@': 'at'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove extra spaces and common words
        text = ' '.join(text.split())
        return text
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two text strings"""
        if pd.isna(text1) or pd.isna(text2):
            return 0.0
        
        norm_text1 = self.normalize_text_for_comparison(text1)
        norm_text2 = self.normalize_text_for_comparison(text2)
        
        if not norm_text1 or not norm_text2:
            return 0.0
        
        return SequenceMatcher(None, norm_text1, norm_text2).ratio()
    
    def find_exact_duplicates(self, df):
        """Find exact duplicates based on key columns"""
        print("   🔍 Finding exact duplicates...")
        
        # Find appropriate columns for duplicate checking
        check_cols = []
        for col in self.duplicate_check_columns:
            if col in df.columns:
                check_cols.append(col)
        
        if not check_cols:
            print("   ⚠️  No suitable columns found for duplicate detection")
            return df, 0
        
        print(f"   📊 Checking duplicates on columns: {check_cols}")
        
        initial_count = len(df)
        
        # Mark duplicates
        df['is_duplicate'] = df.duplicated(subset=check_cols, keep='first')
        
        # Remove exact duplicates
        df_dedup = df[~df['is_duplicate']].copy()
        df_dedup.drop('is_duplicate', axis=1, inplace=True)
        
        duplicates_removed = initial_count - len(df_dedup)
        
        print(f"   ✅ Removed {duplicates_removed} exact duplicates")
        return df_dedup, duplicates_removed
    
    def find_fuzzy_duplicates(self, df, max_comparisons=10000):
        """Find fuzzy duplicates using similarity matching"""
        print("   🔍 Finding fuzzy duplicates...")
        
        # Find job title and company columns
        title_col = None
        company_col = None
        
        for col in ['job_title', 'title', 'position']:
            if col in df.columns:
                title_col = col
                break
        
        for col in ['company', 'company_name']:
            if col in df.columns:
                company_col = col
                break
        
        if not title_col or not company_col:
            print("   ⚠️  Missing job_title or company columns for fuzzy matching")
            return df, 0
        
        print(f"   📊 Using columns: {title_col}, {company_col}")
        
        if len(df) > max_comparisons:
            print(f"   ⚠️  Dataset too large ({len(df)} rows), sampling {max_comparisons} for fuzzy duplicate detection")
            # Sample for performance
            sample_df = df.sample(n=max_comparisons, random_state=42)
            remaining_df = df[~df.index.isin(sample_df.index)]
        else:
            sample_df = df.copy()
            remaining_df = pd.DataFrame()
        
        duplicates_to_remove = set()
        comparisons_made = 0
        max_total_comparisons = 50000  # Limit total comparisons
        
        print(f"   🔄 Comparing {len(sample_df)} records for fuzzy duplicates...")
        
        for i, (idx1, row1) in enumerate(sample_df.iterrows()):
            if comparisons_made >= max_total_comparisons:
                break
                
            for j, (idx2, row2) in enumerate(sample_df.iloc[i+1:].iterrows()):
                comparisons_made += 1
                
                if comparisons_made >= max_total_comparisons:
                    break
                
                # Skip if already marked as duplicate
                if idx2 in duplicates_to_remove:
                    continue
                
                # Calculate similarities
                title_sim = self.calculate_similarity(row1[title_col], row2[title_col])
                company_sim = self.calculate_similarity(row1[company_col], row2[company_col])
                
                # Check if it's a fuzzy duplicate
                if (title_sim >= self.similarity_thresholds['job_title'] and 
                    company_sim >= self.similarity_thresholds['company']):
                    
                    duplicates_to_remove.add(idx2)
                    
                    if comparisons_made % 1000 == 0:
                        print(f"   📊 Processed {comparisons_made} comparisons, found {len(duplicates_to_remove)} fuzzy duplicates")
        
        # Remove fuzzy duplicates from sample
        sample_df_clean = sample_df[~sample_df.index.isin(duplicates_to_remove)]
        
        # Combine with remaining data
        if not remaining_df.empty:
            final_df = pd.concat([sample_df_clean, remaining_df], ignore_index=True)
        else:
            final_df = sample_df_clean.copy()
        
        fuzzy_duplicates_removed = len(duplicates_to_remove)
        print(f"   ✅ Removed {fuzzy_duplicates_removed} fuzzy duplicates")
        
        return final_df, fuzzy_duplicates_removed
    
    def remove_salary_based_duplicates(self, df):
        """Remove duplicates that are identical except for salary"""
        print("   💰 Checking salary-based duplicates...")
        
        # Find core columns (excluding salary columns)
        core_columns = []
        salary_columns = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['salary', 'pay', 'wage', 'compensation']):
                salary_columns.append(col)
            elif col in ['job_title', 'company', 'company_name', 'location', 'job_description']:
                core_columns.append(col)
        
        if len(core_columns) < 2:
            print("   ⚠️  Insufficient core columns for salary-based duplicate detection")
            return df, 0
        
        initial_count = len(df)
        
        # Group by core columns and keep the one with best salary info
        def select_best_salary_record(group):
            if len(group) == 1:
                return group
            
            # Score records based on salary completeness
            group['salary_score'] = 0
            
            for col in salary_columns:
                if col in group.columns:
                    group['salary_score'] += group[col].notna().astype(int)
            
            # Return record with highest salary score
            best_idx = group['salary_score'].idxmax()
            return group.loc[[best_idx]]
        
        print(f"   📊 Grouping by: {core_columns}")
        
        # Group and select best records
        df_grouped = df.groupby(core_columns, as_index=False).apply(select_best_salary_record)
        
        # Reset index
        df_dedup = df_grouped.reset_index(drop=True)
        
        # Remove the salary_score column
        if 'salary_score' in df_dedup.columns:
            df_dedup.drop('salary_score', axis=1, inplace=True)
        
        salary_duplicates_removed = initial_count - len(df_dedup)
        
        print(f"   ✅ Removed {salary_duplicates_removed} salary-based duplicates")
        return df_dedup, salary_duplicates_removed
    
    def deduplicate_dataset(self, filename):
        """Remove duplicates from a single dataset"""
        print(f"\n🗑️  Deduplicating: {filename}")
        
        try:
            # Load merged data
            filepath = self.merged_data_path / filename
            if not filepath.exists():
                print(f"   ❌ File not found: {filename}")
                return None
            
            df = pd.read_csv(filepath, low_memory=False)
            initial_shape = df.shape
            
            print(f"   📊 Initial dataset: {initial_shape[0]:,} rows, {initial_shape[1]} columns")
            
            # Step 1: Remove exact duplicates
            df, exact_dups = self.find_exact_duplicates(df)
            
            # Step 2: Remove fuzzy duplicates (if dataset not too large)
            if len(df) <= 50000:  # Only for manageable datasets
                df, fuzzy_dups = self.find_fuzzy_duplicates(df)
            else:
                print(f"   ⚠️  Skipping fuzzy duplicate detection (dataset too large: {len(df):,} rows)")
                fuzzy_dups = 0
            
            # Step 3: Remove salary-based duplicates
            df, salary_dups = self.remove_salary_based_duplicates(df)
            
            # Add deduplication metadata
            df['deduplication_date'] = '2026-03-04'
            df['exact_duplicates_removed'] = exact_dups
            df['fuzzy_duplicates_removed'] = fuzzy_dups
            df['salary_duplicates_removed'] = salary_dups
            
            # Save deduplicated dataset
            output_filename = filename.replace('merged_', 'deduplicated_')
            output_path = self.deduplicated_data_path / output_filename
            df.to_csv(output_path, index=False)
            
            final_shape = df.shape
            total_removed = initial_shape[0] - final_shape[0]
            
            print(f"   ✅ Final dataset: {final_shape[0]:,} rows, {final_shape[1]} columns")
            print(f"   🗑️  Total removed: {total_removed:,} records ({total_removed/initial_shape[0]*100:.1f}%)")
            print(f"   ✅ Saved: {output_filename}")
            
            return {
                'success': True,
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'exact_duplicates': exact_dups,
                'fuzzy_duplicates': fuzzy_dups,
                'salary_duplicates': salary_dups,
                'total_removed': total_removed,
                'output_file': output_filename
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_all_merged_datasets(self):
        """Process all merged datasets for deduplication"""
        print("🗑️  PHASE 10: DUPLICATE REMOVAL")
        print("="*50)
        
        merged_files = [f for f in os.listdir(self.merged_data_path) 
                       if f.startswith('merged_') and f.endswith('.csv')]
        
        # Also check for master dataset
        master_file = 'maasarthi_master_dataset.csv'
        if (self.merged_data_path / master_file).exists():
            merged_files.append(master_file)
        
        if not merged_files:
            print("❌ No merged datasets found!")
            return {}
        
        dedup_report = {}
        
        for filename in merged_files:
            result = self.deduplicate_dataset(filename)
            if result:
                dedup_report[filename] = result
        
        # Save deduplication report
        report_path = self.deduplicated_data_path / 'deduplication_report.json'
        with open(report_path, 'w') as f:
            json.dump(dedup_report, f, indent=2, default=str)
        
        self.generate_summary(dedup_report)
        return dedup_report
    
    def generate_summary(self, report):
        """Generate deduplication summary"""
        print(f"\n📊 DUPLICATE REMOVAL SUMMARY")
        print("="*50)
        
        successful = len([k for k, v in report.items() if v.get('success', False)])
        failed = len([k for k, v in report.items() if not v.get('success', False)])
        
        print(f"✅ Successfully processed: {successful} datasets")
        print(f"❌ Failed to process: {failed} datasets")
        
        if successful > 0:
            total_initial = sum(v['initial_shape'][0] for v in report.values() if v.get('success'))
            total_final = sum(v['final_shape'][0] for v in report.values() if v.get('success'))
            total_removed = sum(v['total_removed'] for v in report.values() if v.get('success'))
            
            print(f"📊 Total rows: {total_initial:,} → {total_final:,}")
            print(f"🗑️  Total duplicates removed: {total_removed:,} ({total_removed/total_initial*100:.1f}%)")
            
            # Break down by type
            exact_total = sum(v['exact_duplicates'] for v in report.values() if v.get('success'))
            fuzzy_total = sum(v['fuzzy_duplicates'] for v in report.values() if v.get('success'))
            salary_total = sum(v['salary_duplicates'] for v in report.values() if v.get('success'))
            
            print(f"   - Exact duplicates: {exact_total:,}")
            print(f"   - Fuzzy duplicates: {fuzzy_total:,}")
            print(f"   - Salary duplicates: {salary_total:,}")
        
        print(f"\n🎯 READY FOR PHASE 11: Missing data handling")

def main():
    """Main execution function"""
    merged_data_path = Path(__file__).parent.parent / 'Merged_Data'
    deduplicated_data_path = Path(__file__).parent.parent / 'Deduplicated_Data'
    
    if not merged_data_path.exists():
        print("❌ Merged_Data directory not found!")
        print("➡️  Please run phase_09_dataset_merging.py first")
        return
    
    remover = DuplicateRemover(merged_data_path, deduplicated_data_path)
    report = remover.process_all_merged_datasets()
    
    print("\n🎉 Phase 10 Complete: Duplicate removal finished!")
    print("➡️  Next: Run phase_11_missing_data_handling.py")

if __name__ == "__main__":
    main()