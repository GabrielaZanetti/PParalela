/*
 * mandelbrot.cu
 * 
 * RENDERIZADOR DO CONJUNTO DE MANDELBROT COM CUDA
 * 
 * Descrição:
 * Computa o conjunto de Mandelbrot usando paralelismo GPU.
 * Cada thread CUDA calcula a iteração de uma pixel independentemente.
 * O resultado é escrito em um arquivo de imagem PPM (P6 binary).
 * 
 * Uso:
 *   nvcc -O3 mandelbrot.cu -o mandelbrot
 *   ./mandelbrot [width] [height] [maxIter] [output.ppm]
 *   ./mandelbrot 1920 1080 1000 mandelbrot.ppm
 * 
 * Parâmetros padrão:
 *   width=1024, height=768, maxIter=1000
 *   saída: mandelbrot.ppm
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// Verifica erros de CUDA e imprime mensagem de erro se houver
static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Kernel CUDA que computa o número de iterações para cada pixel
// Cada thread calcula um pixel independentemente
// grid(x, y) = pixel(x, y), dimensões até o tamanho da imagem
__global__ void mandelbrot_kernel(unsigned char *img, int w, int h, int maxIter,
                                  double xmin, double xmax, double ymin, double ymax) {
    // Calcula coordenadas globais da thread (pixel)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Ignora threads fora dos limites da imagem
    if (x >= w || y >= h) return;

    // Mapeia coordenadas de pixel para número complexo c = x + iy
    double cx = xmin + (double)x * (xmax - xmin) / (double)(w - 1);
    double cy = ymin + (double)y * (ymax - ymin) / (double)(h - 1);

    // Itera z_{n+1} = z_n^2 + c começando de z_0 = 0
    double zx = 0.0, zy = 0.0;
    int iter = 0;
    while (zx*zx + zy*zy <= 4.0 && iter < maxIter) {
        // Calcula z_new = z^2 + c
        double xt = zx*zx - zy*zy + cx;
        zy = 2.0*zx*zy + cy;
        zx = xt;
        ++iter;
    }

    // Converte número de iterações para cor RGB
    int idx = 3 * (y * w + x);  // Índice do pixel (3 bytes por pixel: R, G, B)
    if (iter == maxIter) {
        // Se atingiu max iterações, ponto está no conjunto: cor preta
        img[idx+0] = img[idx+1] = img[idx+2] = 0;
    } else {
        // Caso contrário, colorir baseado no número de iterações
        // Usa uma escala suavizada com raiz quadrada para melhor visualização
        double t = (double)iter / (double)maxIter;
        unsigned char v = (unsigned char)(255.0 * sqrt(t));
        // Matiz azulada para melhor aparência visual
        img[idx+0] = (unsigned char) (v * 0.6); // Canal vermelho (reduzido)
        img[idx+1] = (unsigned char) (v * 0.9); // Canal verde (médio)
        img[idx+2] = v;                         // Canal azul (completo)
    }

int main(int argc, char **argv) {
    // Parâmetros padrão
    int w = 1024, h = 768, maxIter = 1000;
    const char *out = "mandelbrot.ppm";
    // Região do plano complexo a renderizar
    double xmin = -2.0, xmax = 1.0, ymin = -1.2, ymax = 1.2;

    // Parse argumentos da linha de comando
    if (argc >= 3) { w = atoi(argv[1]); h = atoi(argv[2]); }
    if (argc >= 4) maxIter = atoi(argv[3]);
    if (argc >= 5) out = argv[4];

    // Aloca memória no host (CPU) para a imagem
    size_t imgSize = (size_t)w * (size_t)h * 3;  // 3 bytes (RGB) por pixel
    unsigned char *h_img = (unsigned char*)malloc(imgSize);
    if (!h_img) { fprintf(stderr, "Allocation failed\n"); return 1; }

    // Aloca memória no device (GPU) para a imagem
    unsigned char *d_img;
    checkCuda(cudaMalloc(&d_img, imgSize), "cudaMalloc d_img");

    // Configura grid e blocos de threads
    // Cada bloco tem 16x16 threads (256 threads por bloco)
    // Grade cobre toda a imagem (dimensões de x e y do grid)
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    // Lança kernel CUDA para computar Mandelbrot
    mandelbrot_kernel<<<grid, block>>>(d_img, w, h, maxIter, xmin, xmax, ymin, ymax);
    checkCuda(cudaGetLastError(), "Kernel launch failed");
    
    // Copia resultado do device de volta para o host
    checkCuda(cudaMemcpy(h_img, d_img, imgSize, cudaMemcpyDeviceToHost), "cudaMemcpy d->h");

    // Escreve imagem em formato PPM P6 (formato binário simples)
    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);  // Header PPM
    fwrite(h_img, 1, imgSize, f);           // Dados RGB binários
    fclose(f);

    printf("Wrote %s (%dx%d)\n", out, w, h);

    // Libera memória do device e host
    cudaFree(d_img);
    free(h_img);
    return 0;
}
