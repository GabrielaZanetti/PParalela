#!/usr/bin/env python3
"""
particles_python.py - SIMULAÇÃO DE PARTÍCULAS EM PYTHON

Descrição:
    Simula N partículas sob atração central usando NumPy (CPU).
    Cada partícula sofre aceleração a = -G * r / r^3.
    Integração usando método de Euler explícito.
    Resultado é salvo em formato CSV com posições e velocidades finais.

Uso:
    python particles_python.py [N] [steps] [dt] [output.csv]
    python particles_python.py 20000 500 0.01 particles.csv

Parâmetros padrão:
    N=20000, steps=200, dt=0.01
    saída: particles.csv

Performance:
    - Vetorizado com NumPy (CPU), mais lento que CUDA
    - Sem dependência de NVIDIA GPU
"""

import numpy as np
import sys

def simulate(N, steps, dt, damping=0.999, G=1.0):
    """
    Simula N partículas sob atração central por 'steps' passos de tempo.
    
    Args:
        N: número de partículas
        steps: número de passos de integração
        dt: tamanho do passo de tempo
        damping: coeficiente de amortecimento (redução de velocidade)
        G: constante de atração
    
    Returns:
        pos: array (N, 3) com posições finais
        vel: array (N, 3) com velocidades finais
    """
    # Inicializa posições aleatórias no cubo [-1, 1]^3
    pos = np.random.uniform(-1, 1, (N, 3))
    
    # Inicializa velocidades nulas
    vel = np.zeros((N, 3))
    
    # Loop de integração
    for _ in range(steps):
        # Calcula distância ao quadrado (r^2) com pequeno epsilon para evitar divisão por zero
        r2 = np.sum(pos**2, axis=1, keepdims=True) + 1e-6
        
        # Calcula 1/r^3 = (r^2)^(-1.5)
        r_inv3 = r2 ** (-1.5)
        
        # Calcula aceleração: a = -G * pos / r^3
        acc = -G * pos * r_inv3
        
        # Integra velocidade: v_novo = v_old + a * dt
        vel += acc * dt
        
        # Integra posição: r_novo = r_old + v * dt
        pos += vel * dt
        
        # Aplica amortecimento: v *= damping
        vel *= damping
    
    return pos, vel

def save_csv(filename, pos, vel):
    """
    Escreve posições e velocidades das partículas em arquivo CSV.
    
    Formato CSV:
        id,x,y,z,vx,vy,vz
        0,x0,y0,z0,vx0,vy0,vz0
        1,x1,y1,z1,vx1,vy1,vz1
        ...
    
    Args:
        filename: nome do arquivo de saída
        pos: array (N, 3) com posições
        vel: array (N, 3) com velocidades
    """
    with open(filename, 'w') as f:
        # Escreve header do CSV
        f.write("id,x,y,z,vx,vy,vz\n")
        # Escreve dados de cada partícula
        for i in range(len(pos)):
            f.write(f"{i},{pos[i,0]:.6f},{pos[i,1]:.6f},{pos[i,2]:.6f},"
                   f"{vel[i,0]:.6f},{vel[i,1]:.6f},{vel[i,2]:.6f}\n")
    print(f"Wrote {filename} (N={len(pos)}, steps={steps})")

if __name__ == "__main__":
    # Define seed para reprodutibilidade
    np.random.seed(1234)
    
    # Parâmetros padrão
    N, steps, dt = 20000, 200, 0.01
    out = "particles.csv"
    
    # Parse argumentos da linha de comando
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    if len(sys.argv) >= 3:
        steps = int(sys.argv[2])
    if len(sys.argv) >= 4:
        dt = float(sys.argv[3])
    if len(sys.argv) >= 5:
        out = sys.argv[4]
    
    # Executa simulação
    pos, vel = simulate(N, steps, dt)
    
    # Salva resultados em CSV
    save_csv(out, pos, vel)
