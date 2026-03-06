#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MAASARTHI - TRAIN ALL ML MODELS                           ║
║                  Master Script to Train All 8 Models                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Model configurations
MODELS = [
    {
        'name': 'Job Recommendation',
        'script': 'models/job_recommendation_model.py',
        'output': 'job_recommendation_model.pkl'
    },
    {
        'name': 'Income Prediction',
        'script': 'models/income_prediction_model.py',
        'output': 'income_prediction_model.pkl'
    },
    {
        'name': 'Skill-Job Matching',
        'script': 'models/skill_job_matching_model.py',
        'output': 'skill_job_matching_model.pkl'
    },
    {
        'name': 'Mother Suitability',
        'script': 'models/mother_suitability_model.py',
        'output': 'mother_suitability_model.pkl'
    },
    {
        'name': 'Skill Gap Analyzer',
        'script': 'models/skill_gap_analyzer.py',
        'output': 'skill_gap_analyzer_model.pkl'
    },
    {
        'name': 'Career Path Predictor',
        'script': 'models/career_path_predictor.py',
        'output': 'career_path_model.pkl'
    },
    {
        'name': 'Work-Life Balance',
        'script': 'models/work_life_balance_model.py',
        'output': 'work_life_balance_model.pkl'
    },
    {
        'name': 'Profile Completeness',
        'script': 'models/profile_completeness_scorer.py',
        'output': 'profile_completeness_model.pkl'
    }
]


def train_model(model_info, sample_frac=1.0):
    """Train a single model"""
    print(f"\n{'='*70}")
    print(f"🚀 Training: {model_info['name']}")
    print(f"{'='*70}")
    print(f"   Script: {model_info['script']}")
    print(f"   Sample: {sample_frac*100:.0f}%")
    print(f"   Started: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    cmd = [sys.executable, model_info['script']]
    if sample_frac < 1.0:
        cmd.extend(['--sample', str(sample_frac)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per model
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n   ✅ {model_info['name']} - SUCCESS")
            print(f"   ⏱️  Time: {elapsed/60:.1f} minutes")
            
            # Extract accuracy from output if possible
            for line in result.stdout.split('\n'):
                if 'accuracy' in line.lower() or 'Accuracy' in line:
                    print(f"   {line.strip()}")
            
            return True, elapsed
        else:
            print(f"\n   ❌ {model_info['name']} - FAILED")
            print(f"   Error: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
            return False, elapsed
            
    except subprocess.TimeoutExpired:
        print(f"\n   ⏰ {model_info['name']} - TIMEOUT (>2 hours)")
        return False, 7200
    except Exception as e:
        print(f"\n   ❌ {model_info['name']} - ERROR: {str(e)}")
        return False, 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all MaaSarthi ML models')
    parser.add_argument('--sample', type=float, default=1.0,
                       help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated model indices (1-8) or "all"')
    parser.add_argument('--skip', type=str, default='',
                       help='Comma-separated model indices to skip')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎯 MAASARTHI - TRAINING ALL ML MODELS")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Sample fraction: {args.sample*100:.0f}%")
    print(f"   Total models: {len(MODELS)}")
    
    # Parse model selection
    if args.models.lower() == 'all':
        selected_indices = list(range(len(MODELS)))
    else:
        selected_indices = [int(i)-1 for i in args.models.split(',')]
    
    # Parse skip list
    skip_indices = []
    if args.skip:
        skip_indices = [int(i)-1 for i in args.skip.split(',')]
    
    # Filter models
    models_to_train = [
        (i, MODELS[i]) for i in selected_indices 
        if i not in skip_indices and i < len(MODELS)
    ]
    
    print(f"   Models to train: {len(models_to_train)}")
    for i, m in models_to_train:
        print(f"      {i+1}. {m['name']}")
    
    # Train each model
    results = []
    total_start = time.time()
    
    for idx, model in models_to_train:
        success, elapsed = train_model(model, args.sample)
        results.append({
            'name': model['name'],
            'success': success,
            'time': elapsed
        })
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n   Total models: {len(results)}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
    
    print("\n   Results:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"      {status} {r['name']}: {r['time']/60:.1f} min")
    
    print(f"\n   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check trained models
    model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    if os.path.exists(model_dir):
        print(f"\n   Trained models in {model_dir}:")
        for f in sorted(os.listdir(model_dir)):
            if f.endswith('.pkl'):
                size = os.path.getsize(os.path.join(model_dir, f)) / (1024*1024)
                print(f"      📦 {f}: {size:.1f} MB")


if __name__ == "__main__":
    main()
