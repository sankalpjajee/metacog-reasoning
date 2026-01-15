#!/usr/bin/env python3
"""
Test script for MR-Ben benchmark integration
"""

import sys
sys.path.insert(0, '/home/jajee/metacog-reasoning/src')

from evaluation.benchmarks import load_benchmark

def test_mrben():
    """Test loading MR-Ben benchmark."""
    print("Testing MR-Ben benchmark loading...")
    print("="*60)
    
    try:
        # Try to load a small sample
        samples = load_benchmark('mrben', split='test')
        
        print(f"✓ Successfully loaded {len(samples)} samples from MR-Ben")
        print("\nFirst sample:")
        print(f"  ID: {samples[0].id}")
        print(f"  Category: {samples[0].category}")
        print(f"  Difficulty: {samples[0].difficulty}")
        print(f"  Question: {samples[0].question[:200]}...")
        print(f"  Answer: {samples[0].answer}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading MR-Ben: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mrben()
    sys.exit(0 if success else 1)
