#!/usr/bin/env python3
"""
visualize.py - Visualize Mandelbrot PPM and particle CSV outputs
"""

import sys
import numpy as np
from pathlib import Path

def visualize_ppm(filename):
    """Display PPM image using matplotlib"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return
    
    if not Path(filename).exists():
        print(f"File not found: {filename}")
        return
    
    # Read PPM
    with open(filename, 'rb') as f:
        line = f.readline().decode('ascii').strip()
        if line != "P6":
            print(f"ERROR: Expected P6, got {line}")
            return
        w, h = map(int, f.readline().decode('ascii').strip().split())
        maxval = int(f.readline().decode('ascii').strip())
        img_data = f.read()
    
    img = np.frombuffer(img_data, dtype=np.uint8).reshape((h, w, 3))
    
    plt.figure(figsize=(12, 9))
    plt.imshow(img)
    plt.title("Mandelbrot Set")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_particles(filename):
    """Plot particle positions in 3D"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return
    
    if not Path(filename).exists():
        print(f"File not found: {filename}")
        return
    
    # Read CSV
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    if data.size == 0:
        print("No data in CSV")
        return
    
    pos = data[:, 1:4]  # x, y, z columns
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Particle Positions')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize.py <mandelbrot.ppm>")
        print("  python visualize.py <particles.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if filename.endswith('.ppm'):
        visualize_ppm(filename)
    elif filename.endswith('.csv'):
        visualize_particles(filename)
    else:
        print(f"Unknown file type: {filename}")
