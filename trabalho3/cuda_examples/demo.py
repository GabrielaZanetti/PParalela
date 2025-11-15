#!/usr/bin/env python3
"""
demo.py - Run all examples and optionally visualize
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, desc):
    """Run command and report status"""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {desc} completed successfully")
            return True
        else:
            print(f"✗ {desc} failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    py = sys.executable
    
    print("CUDA Examples Demo")
    print("=" * 60)
    
    # Mandelbrot
    print("\n[1/2] Running Mandelbrot example...")
    run_cmd(f'{py} mandelbrot_python.py 1024 768 1000 mandelbrot_demo.ppm',
            'Mandelbrot (1024x768, 1000 iterations)')
    
    # Particles
    print("\n[2/2] Running Particles example...")
    run_cmd(f'{py} particles_python.py 10000 500 0.01 particles_demo.csv',
            'Particles (10000 particles, 500 steps)')
    
    # Check outputs
    print("\n" + "="*60)
    print("  Output Files")
    print("="*60)
    
    for f in ['mandelbrot_demo.ppm', 'particles_demo.csv']:
        p = Path(f)
        if p.exists():
            size = p.stat().st_size / 1024
            print(f"✓ {f} ({size:.1f} KB)")
        else:
            print(f"✗ {f} not found")
    
    # Optional visualization
    print("\n" + "="*60)
    try:
        import matplotlib
        print("Matplotlib available. Visualizing...")
        run_cmd(f'{py} visualize.py mandelbrot_demo.ppm', 'Visualizing Mandelbrot')
        run_cmd(f'{py} visualize.py particles_demo.csv', 'Visualizing Particles')
    except ImportError:
        print("Matplotlib not installed. To visualize:")
        print("  pip install matplotlib")
        print("  python visualize.py mandelbrot_demo.ppm")
        print("  python visualize.py particles_demo.csv")
