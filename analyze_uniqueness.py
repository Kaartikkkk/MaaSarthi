"""Analyze uniqueness of all datasets"""
import pandas as pd

print("="*70)
print("DATASET UNIQUENESS ANALYSIS")
print("="*70)

# Load the job titles file
job_titles_df = pd.read_csv('job_titles_11792.csv')
print(f"\n1. job_titles_11792.csv:")
print(f"   Total titles: {len(job_titles_df)}")
print(f"   Unique titles: {job_titles_df['job_title'].nunique()}")

# Load current datasets
print(f"\n2. Current Datasets:")

# Original dataset
try:
    orig = pd.read_csv('dataset.csv')
    print(f"   dataset.csv: {len(orig)} records")
    for col in orig.columns:
        print(f"     {col}: {orig[col].nunique()} unique")
except Exception as e:
    print(f"   dataset.csv: Error - {e}")

# 300k v2 dataset
try:
    v2 = pd.read_csv('maasarthi_300k_v2_dataset.csv')
    print(f"\n   maasarthi_300k_v2_dataset.csv: {len(v2)} records")
    print(f"     Unique job titles: {v2['job_title'].nunique()}")
    print(f"     Unique domains: {v2['domain'].nunique()}")
    print(f"     Unique companies: {v2['company'].nunique()}")
    print(f"     Unique cities: {v2['city'].nunique()}")
    print(f"     Unique education: {v2['education'].nunique()}")
except Exception as e:
    print(f"   maasarthi_300k_v2_dataset.csv: Error - {e}")

# Show sample from job_titles file
print(f"\n3. Sample Job Titles from job_titles_11792.csv (20 random):")
for i, title in enumerate(job_titles_df['job_title'].sample(20, random_state=42).values):
    print(f"   {i+1}. {title}")

print("\n" + "="*70)
