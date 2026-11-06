"""Compatibility entry point for the corrected three-seed comparison.

The old one-seed A comparison is archived under results/lei/available_comparison.
Run python -m analysis.compare_model_b for the current reproducible analysis.
"""
from analysis.compare_model_b import main

if __name__ == "__main__":
    main()
