"""
Quick Launcher for Visual Comparison and Heatmap Generator.
Run this script directly from terminal or IDE:
    python scripts/visualize_comparison.py
"""
import subprocess
import sys
import os

if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    target_script = os.path.join(root_dir, 'scripts', 'visualization', 'visualize_stereo_comparison.py')
    cmd = [sys.executable, target_script] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
