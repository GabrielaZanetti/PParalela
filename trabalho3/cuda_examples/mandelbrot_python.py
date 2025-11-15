#!/usr/bin/env python3
"""
mandelbrot_python.py - Mandelbrot set renderer (Python + NumPy)
Outputs a PPM image file.
"""

import numpy as np
import sys
from pathlib import Path

def mandelbrot(w, h, max_iter, xmin, xmax, ymin, ymax):
    """Compute Mandelbrot set"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(h):
        cy = ymin + (y / (h - 1)) * (ymax - ymin)
        for x in range(w):
            cx = xmin + (x / (w - 1)) * (xmax - xmin)
            
            zx, zy = 0.0, 0.0
            for it in range(max_iter):
                if zx*zx + zy*zy > 4.0:
                    break
                xt = zx*zx - zy*zy + cx
                zy = 2.0*zx*zy + cy
                zx = xt
            else:
                it = max_iter - 1
            
            # Coloring
            if it == max_iter - 1:
                img[y, x] = [0, 0, 0]
            else:
                t = it / max_iter
                v = int(255.0 * np.sqrt(t))
                img[y, x] = [int(v * 0.6), int(v * 0.9), v]
    
    return img

def write_ppm(filename, img):
    """Write PPM P6 format"""
    h, w, _ = img.shape
    with open(filename, 'wb') as f:
        f.write(f"P6\n{w} {h}\n255\n".encode('ascii'))
        f.write(img.tobytes())
    print(f"Wrote {filename} ({w}x{h})")

if __name__ == "__main__":
    w, h = 1024, 768
    max_iter = 1000
    out = "mandelbrot.ppm"
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.2, 1.2
    
    if len(sys.argv) >= 3:
        w, h = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) >= 4:
        max_iter = int(sys.argv[3])
    if len(sys.argv) >= 5:
        out = sys.argv[4]
    
    img = mandelbrot(w, h, max_iter, xmin, xmax, ymin, ymax)
    write_ppm(out, img)
