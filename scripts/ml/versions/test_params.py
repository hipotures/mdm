#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

# Add path to import from version_3
sys.path.insert(0, str(Path(__file__).parent))

def test_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-folds', type=int, default=3)
    parser.add_argument('--removal-ratio', type=float, default=0.1)
    parser.add_argument('--tuning-trials', type=int, default=20)
    parser.add_argument('--no-tuning', action='store_true')
    parser.add_argument('--random-state', type=int, default=42)
    
    # Test with custom parameters
    test_args = ['--cv-folds', '5', '--removal-ratio', '0.2', '--no-tuning', '--tuning-trials', '50']
    args = parser.parse_args(test_args)
    
    print(f"cv_folds: {args.cv_folds}")
    print(f"removal_ratio: {args.removal_ratio}")
    print(f"tuning_trials: {args.tuning_trials}")
    print(f"no_tuning: {args.no_tuning}")
    print(f"use_tuning: {not args.no_tuning}")
    print(f"random_state: {args.random_state}")

if __name__ == '__main__':
    test_params()