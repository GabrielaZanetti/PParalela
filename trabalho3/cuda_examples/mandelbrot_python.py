#!/usr/bin/env python3
"""
RENDERIZADOR DO CONJUNTO DE MANDELBROT EM PYTHON

Renderiza o conjunto de Mandelbrot usando NumPy (CPU).
Cada pixel é mapeado para um número complexo c;
iteramos z_{n+1} = z_n^2 + c até divergir ou atingir max iterações.
Resultado é salvo em formato de imagem PPM.

Uso:
    python mandelbrot_python.py [width] [height] [maxIter] [output.ppm]
    python mandelbrot_python.py 1920 1080 1000 mandelbrot.ppm

Parâmetros padrão:
    width=1024, height=768, maxIter=1000
    saída: mandelbrot.ppm

Performance:
    - Mais lento que CUDA (CPU puro), mas portável
    - Não requer NVIDIA GPU ou compilação
"""

import numpy as np
import sys
from pathlib import Path

def mandelbrot(w, h, max_iter, xmin, xmax, ymin, ymax):
    """
    Computa o conjunto de Mandelbrot iterando z = z^2 + c para cada pixel.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(h):
        # Mapeia coordenada de pixel para componente imaginária
        cy = ymin + (y / (h - 1)) * (ymax - ymin)
        for x in range(w):
            # Mapeia coordenada de pixel para componente real
            cx = xmin + (x / (w - 1)) * (xmax - xmin)
            
            # Itera z_{n+1} = z_n^2 + c até divergir ou atingir max
            zx, zy = 0.0, 0.0
            for it in range(max_iter):
                # Verifica condição de divergência
                if zx*zx + zy*zy > 4.0:
                    break
                # Calcula z_novo = z^2 + c
                xt = zx*zx - zy*zy + cx
                zy = 2.0*zx*zy + cy
                zx = xt
            else:
                # Se não divergiu, it = max_iter - 1 (ponto no conjunto)
                it = max_iter - 1
            
            # Converte número de iterações para cor RGB
            if it == max_iter - 1:
                # Ponto no conjunto: preto
                img[y, x] = [0, 0, 0]
            else:
                # Pontos fora: cor baseada em iterações
                t = it / max_iter
                v = int(255.0 * np.sqrt(t))
                img[y, x] = [int(v * 0.6), int(v * 0.9), v]
    
    return img

def write_ppm(filename, img):
    """
    Escreve imagem em formato PPM P6
    """
    h, w, _ = img.shape
    with open(filename, 'wb') as f:
        # Escreve header PPM
        f.write(f"P6\n{w} {h}\n255\n".encode('ascii'))
        # Escreve dados RGB binários
        f.write(img.tobytes())
    print(f"Wrote {filename} ({w}x{h})")

if __name__ == "__main__":
    # Parâmetros padrão
    w, h = 1024, 768
    max_iter = 1000
    out = "mandelbrot.ppm"
    # Região padrão do plano complexo
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.2, 1.2
    
    # Parse argumentos da linha de comando
    if len(sys.argv) >= 3:
        w, h = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) >= 4:
        max_iter = int(sys.argv[3])
    if len(sys.argv) >= 5:
        out = sys.argv[4]
    
    # Computa o conjunto de Mandelbrot
    img = mandelbrot(w, h, max_iter, xmin, xmax, ymin, ymax)
    
    # Escreve resultado em arquivo PPM
    write_ppm(out, img)
