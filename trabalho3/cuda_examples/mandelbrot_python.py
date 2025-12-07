import numpy as np
import sys
import time  # para medir o tempo
from pathlib import Path

def mandelbrot(w, h, max_iter, xmin, xmax, ymin, ymax):
    """
    Calcula o conjunto de Mandelbrot para cada pixel da imagem.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(h):
        cy = ymin + (y / (h - 1)) * (ymax - ymin)  # mapeia a coordenada y para o valor imaginário
        for x in range(w):
            cx = xmin + (x / (w - 1)) * (xmax - xmin)  # mapeia a coordenada x para o valor real
            
            zx, zy = 0.0, 0.0  # inicializa z
            for it in range(max_iter):
                if zx*zx + zy*zy > 4.0:  # verifica se o ponto diverge
                    break
                xt = zx*zx - zy*zy + cx
                zy = 2.0*zx*zy + cy
                zx = xt
            else:
                it = max_iter - 1  # se não divergiu, o ponto está no conjunto
            
            if it == max_iter - 1:
                img[y, x] = [0, 0, 0]  # ponto no conjunto é preto
            else:
                t = it / max_iter  # calcula cor para os pontos fora do conjunto
                v = int(255.0 * np.sqrt(t))
                img[y, x] = [int(v * 0.6), int(v * 0.9), v]
    
    return img

def write_ppm(filename, img):
    """
    Salva a imagem no formato PPM.
    """
    h, w, _ = img.shape
    with open(filename, 'wb') as f:
        f.write(f"P6\n{w} {h}\n255\n".encode('ascii'))  # escreve o cabeçalho PPM
        f.write(img.tobytes())  # escreve os dados RGB

if __name__ == "__main__":
    # parâmetros padrão
    w, h = 1024, 768
    max_iter = 1000
    out = "mandelbrot.ppm"
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.2, 1.2
    
    # le parâmetros da linha de comando
    if len(sys.argv) >= 3:
        w, h = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) >= 4:
        max_iter = int(sys.argv[3])
    if len(sys.argv) >= 5:
        out = sys.argv[4]

    # medir tempo de execução 
    start_time_cpu = time.time()
    
    # gera a imagem do Mandelbrot 
    img_cpu = mandelbrot(w, h, max_iter, xmin, xmax, ymin, ymax)
    
    end_time_cpu = time.time()
    execution_time_ms = (end_time_cpu - start_time_cpu) * 1000
    
    # salva o resultado 
    write_ppm(out, img_cpu)
    
    print(f"{execution_time_ms:.0f} ms")
