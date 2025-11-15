#!/usr/bin/env python3
"""
convert_ppm.py - Convert PPM images to PNG format
"""

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Install with:")
    print("  pip install pillow")
    sys.exit(1)

def convert_ppm_to_png(ppm_file, png_file=None):
    """Convert PPM to PNG"""
    ppm_path = Path(ppm_file)
    
    if not ppm_path.exists():
        print(f"ERROR: File not found: {ppm_file}")
        return False
    
    if png_file is None:
        png_file = ppm_path.stem + ".png"
    
    try:
        print(f"Converting {ppm_file} -> {png_file}...", end=" ", flush=True)
        img = Image.open(ppm_file)
        img.save(png_file, "PNG")
        
        ppm_size = ppm_path.stat().st_size / 1024 / 1024
        png_size = Path(png_file).stat().st_size / 1024 / 1024
        
        print(f"✓ Done!")
        print(f"  PPM: {ppm_size:.2f} MB  ->  PNG: {png_size:.2f} MB")
        print(f"  Compression: {100 * (1 - png_size/ppm_size):.1f}%")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_ppm.py <input.ppm> [output.png]")
        print("\nExample:")
        print("  python convert_ppm.py mandelbrot.ppm")
        print("  python convert_ppm.py mandelbrot.ppm mandelbrot_hd.png")
        sys.exit(1)
    
    ppm_file = sys.argv[1]
    png_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if convert_ppm_to_png(ppm_file, png_file):
        sys.exit(0)
    else:
        sys.exit(1)
