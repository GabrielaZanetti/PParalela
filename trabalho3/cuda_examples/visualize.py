#!/usr/bin/env python3
"""
visualize.py - VISUALIZADOR DE SAÍDAS (PPM e CSV)

Descrição:
    Visualiza os resultados gerados pelos exemplos Mandelbrot e partículas.
    - visualize_ppm: exibe imagem do Mandelbrot usando matplotlib
    - visualize_particles: cria gráfico 3D interativo das posições de partículas

Uso:
    python visualize.py mandelbrot_demo.ppm      # Mostrar imagem Mandelbrot
    python visualize.py particles_demo.csv       # Mostrar partículas em 3D

Dependências:
    matplotlib, numpy
    Instale com: pip install matplotlib numpy
"""

import sys
import numpy as np
from pathlib import Path

def visualize_ppm(filename):
    """
    Exibe imagem PPM usando matplotlib.
    
    Args:
        filename: caminho do arquivo PPM a visualizar
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERRO: matplotlib não instalado. Execute: pip install matplotlib")
        return
    
    if not Path(filename).exists():
        print(f"Arquivo não encontrado: {filename}")
        return
    
    # Lê arquivo PPM (formato P6 binário)
    with open(filename, 'rb') as f:
        # Lê magic number (deve ser "P6")
        line = f.readline().decode('ascii').strip()
        if line != "P6":
            print(f"ERRO: Esperava P6, obteve {line}")
            return
        # Lê dimensões (largura e altura)
        w, h = map(int, f.readline().decode('ascii').strip().split())
        # Lê valor máximo (255)
        maxval = int(f.readline().decode('ascii').strip())
        # Lê dados RGB binários
        img_data = f.read()
    
    # Converte bytes para array NumPy com shape (height, width, 3)
    img = np.frombuffer(img_data, dtype=np.uint8).reshape((h, w, 3))
    
    # Exibe imagem usando matplotlib
    plt.figure(figsize=(12, 9))
    plt.imshow(img)
    plt.title("Conjunto de Mandelbrot")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_particles(filename):
    """
    Cria gráfico 3D interativo das posições de partículas.
    
    Args:
        filename: caminho do arquivo CSV contendo dados das partículas
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("ERRO: matplotlib não instalado. Execute: pip install matplotlib")
        return
    
    if not Path(filename).exists():
        print(f"Arquivo não encontrado: {filename}")
        return
    
    # Lê dados CSV (colunas: id, x, y, z, vx, vy, vz)
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    if data.size == 0:
        print("Nenhum dado no arquivo CSV")
        return
    
    # Extrai colunas de posição (x, y, z)
    pos = data[:, 1:4]
    # Cria figura 3D e scatter plot das posições
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Posições das Partículas')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python visualize.py <mandelbrot.ppm>   # Visualizar Mandelbrot")
        print("  python visualize.py <particles.csv>    # Visualizar partículas em 3D")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Detecta tipo de arquivo pela extensão
    if filename.endswith('.ppm'):
        visualize_ppm(filename)
    elif filename.endswith('.csv'):
        visualize_particles(filename)
    else:
        print(f"Tipo de arquivo desconhecido: {filename}")
