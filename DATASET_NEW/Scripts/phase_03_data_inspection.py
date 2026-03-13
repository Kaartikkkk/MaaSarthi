"""
MAASARTHI DATA PROCESSING PIPELINE
=================================
PHASE 3: DATASET INSPECTION
=================================

This script systematically inspects all raw datasets to understand:
- Column names and data types
- Dataset sizes and shapes
- Missing values and data quality
- Sample data for each dataset

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

class DatasetInspector:
    def __init__(self, raw_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.inspection_report = {}
        
    def inspect_dataset(self, filename):
        """Inspect a single dataset and return comprehensive information"""
        filepath = self.raw_data_path / filename
        
        try:
            # Load dataset
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath, low_memory=False)
            else:
                return None
                
            # Basic information
            info = {
                'filename': filename,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB",
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
                'sample_data': df.head(3).to_dict('records'),
                'unique_counts': {col: df[col].nunique() if df[col].dtype in ['object', 'int64', 'float64'] else 'N/A' 
                                for col in df.columns}
            }
            
            # Additional statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                info['numeric_stats'] = df[numeric_cols].describe().to_dict()
            
            # Additional statistics for text columns
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                info['text_stats'] = {
                    col: {
                        'unique_count': df[col].nunique(),
                        'most_common': df[col].value_counts().head(3).to_dict() if df[col].nunique() < 1000 else 'Too many unique values'
                    }
                    for col in text_cols
                }
            
            return info
            
        except Exception as e:
            return {
                'filename': filename,
                'error': str(e),
                'status': 'FAILED'
            }
    
    def inspect_all_datasets(self):
        """Inspect all CSV files in the raw data directory"""
        print("🔍 PHASE 3: DATASET INSPECTION")
        print("="*50)
        
        csv_files = [f for f in os.listdir(self.raw_data_path) if f.endswith('.csv')]
        
        for filename in csv_files:
            print(f"\n📊 Inspecting: {filename}")
            info = self.inspect_dataset(filename)
            
            if info and 'error' not in info:
                self.inspection_report[filename] = info
                print(f"   ✅ Shape: {info['shape']}")
                print(f"   ✅ Columns: {len(info['columns'])}")
                print(f"   ✅ Memory: {info['memory_usage']}")
                
                # Show columns with missing values
                missing_cols = {k: v for k, v in info['missing_percentage'].items() if v > 0}
                if missing_cols:
                    print(f"   ⚠️  Missing values: {len(missing_cols)} columns")
                else:
                    print(f"   ✅ No missing values")
            else:
                print(f"   ❌ Error: {info.get('error', 'Unknown error')}")
        
        self.save_inspection_report()
        self.generate_summary()
        
    def save_inspection_report(self):
        """Save detailed inspection report to JSON"""
        output_file = self.raw_data_path.parent / 'Scripts' / 'inspection_report.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.inspection_report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed inspection report saved: {output_file}")
    
    def generate_summary(self):
        """Generate and display summary statistics"""
        print("\n📋 INSPECTION SUMMARY")
        print("="*50)
        
        total_datasets = len(self.inspection_report)
        total_rows = sum(info['shape'][0] for info in self.inspection_report.values())
        total_columns = sum(len(info['columns']) for info in self.inspection_report.values())
        
        print(f"📁 Total Datasets: {total_datasets}")
        print(f"📊 Total Rows: {total_rows:,}")
        print(f"📋 Total Columns: {total_columns}")
        
        # Dataset size ranking
        print(f"\n🏆 LARGEST DATASETS:")
        sorted_datasets = sorted(self.inspection_report.items(), 
                               key=lambda x: x[1]['shape'][0], reverse=True)
        
        for i, (filename, info) in enumerate(sorted_datasets[:10], 1):
            print(f"   {i:2d}. {filename:<30} {info['shape'][0]:>8,} rows")
        
        # Datasets suitable for MaaSarthi
        print(f"\n🎯 MAASARTHI-RELEVANT DATASETS:")
        relevant_keywords = ['job', 'company', 'skill', 'salary', 'india', 'gender', 'employee']
        
        for filename, info in self.inspection_report.items():
            if any(keyword in filename.lower() for keyword in relevant_keywords):
                print(f"   ✅ {filename:<30} {info['shape'][0]:>8,} rows - {info['memory_usage']}")
        
        return {
            'total_datasets': total_datasets,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'largest_datasets': sorted_datasets[:10]
        }

def main():
    """Main execution function"""
    raw_data_path = Path(__file__).parent.parent / 'Raw_Data'
    
    if not raw_data_path.exists():
        print("❌ Raw_Data directory not found!")
        return
    
    inspector = DatasetInspector(raw_data_path)
    inspector.inspect_all_datasets()
    
    print("\n🎉 Phase 3 Complete: Dataset inspection finished!")
    print("➡️  Next: Run phase_04_column_standardization.py")

if __name__ == "__main__":
    main()