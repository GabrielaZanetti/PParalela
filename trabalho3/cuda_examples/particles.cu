/*
 * particles.cu
 * 
 * SIMULAÇÃO DE PARTÍCULAS COM CUDA
 * 
 * Descrição:
 * Simula N partículas em um campo de força central (atração à origem).
 * Cada thread CUDA integra uma partícula usando método de Euler.
 * Posições e velocidades finais são salvos em arquivo CSV.
 * 
 * Física:
 *   a = -G * r / (r^3 + epsilon)  [aceleração por atração central]
 *   v = v_old + a * dt            [integração da velocidade]
 *   r = r_old + v * dt            [integração da posição]
 *   v *= damping                  [amortecimento/dissipação]
 * 
 * Uso:
 *   nvcc -O3 particles.cu -o particles
 *   ./particles [N] [steps] [dt] [output.csv]
 *   ./particles 20000 500 0.01 particles.csv
 * 
 * Parâmetros padrão:
 *   N=20000, steps=200, dt=0.01
 *   saída: particles.csv
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Verifica erros de CUDA e imprime mensagem se houver
static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Estrutura para armazenar posição e velocidade de uma partícula
struct Particle { float3 pos; float3 vel; };

// Kernel CUDA que integra N partículas em um passo de tempo
// Cada thread integra uma partícula independentemente
__global__ void integrate(Particle *p, int n, float dt, float damping, float G) {
    // Índice global da thread (ID da partícula)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // Ignora threads fora do range
    
    // Lê posição e velocidade da partícula
    float3 pos = p[i].pos;
    float3 vel = p[i].vel;

    // Calcula aceleração com atração central: a = -G * r / r^3
    // r^2 = x^2 + y^2 + z^2
    float r2 = pos.x*pos.x + pos.y*pos.y + pos.z*pos.z + 1e-6f;  // epsilon para evitar divisão por zero
    float invr = rsqrtf(r2);      // 1/sqrt(r^2)
    float invr3 = invr * invr * invr;  // 1/r^3
    
    // Componentes da aceleração: a_i = -G * pos_i / r^3
    float3 acc;
    acc.x = -G * pos.x * invr3;
    acc.y = -G * pos.y * invr3;
    acc.z = -G * pos.z * invr3;

    // Integra velocidade: v_new = v_old + a * dt
    vel.x += acc.x * dt;
    vel.y += acc.y * dt;
    vel.z += acc.z * dt;

    // Integra posição: r_new = r_old + v * dt
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    // Aplica amortecimento (dissipação de energia): v *= damping
    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;

    // Escreve posição e velocidade atualizadas na memória global
    p[i].pos = pos;
    p[i].vel = vel;
}

int main(int argc, char **argv) {
    // Parâmetros padrão
    int N = 20000;        // Número de partículas
    int steps = 200;      // Número de passos de integração
    float dt = 0.01f;     // Tamanho do passo de tempo
    const char *out = "particles.csv";  // Arquivo de saída

    // Parse argumentos da linha de comando
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) steps = atoi(argv[2]);
    if (argc >= 4) dt = atof(argv[3]);
    if (argc >= 5) out = argv[4];

    // Aloca memória no host (CPU) para as partículas
    Particle *h_p = (Particle*)malloc(sizeof(Particle) * N);
    if (!h_p) { fprintf(stderr, "host alloc failed\n"); return 1; }

    // Inicializa partículas com posições aleatórias em uma esfera unitária
    srand(1234);  // Seed fixo para reprodutibilidade
    for (int i = 0; i < N; ++i) {
        // Gera coordenadas aleatórias no cubo [-1, 1]^3
        float u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float w = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_p[i].pos = make_float3(u, v, w);
        h_p[i].vel = make_float3(0.0f, 0.0f, 0.0f);  // Velocidade inicial nula
    }

    // Aloca memória no device (GPU) para as partículas
    Particle *d_p;
    checkCuda(cudaMalloc(&d_p, sizeof(Particle) * N), "cudaMalloc d_p");
    
    // Copia dados das partículas do host para o device
    checkCuda(cudaMemcpy(d_p, h_p, sizeof(Particle) * N, cudaMemcpyHostToDevice), "cudaMemcpy h->d");

    // Configura grid e blocos de threads
    // 256 threads por bloco (bom para utilização de GPU)
    int block = 256;
    int grid = (N + block - 1) / block;  // Cálculo do número de blocos

    // Parâmetros da simulação
    float damping = 0.999f;  // Amortecimento (dissipação de energia)
    float G = 1.0f;          // Constante de atração

    // Loop de integração: executa N passos de tempo
    for (int s = 0; s < steps; ++s) {
        integrate<<<grid, block>>>(d_p, N, dt, damping, G);
        checkCuda(cudaGetLastError(), "kernel launch integrate");
    }

    // Copia resultado final do device de volta para o host
    checkCuda(cudaMemcpy(h_p, d_p, sizeof(Particle) * N, cudaMemcpyDeviceToHost), "cudaMemcpy d->h");

    // Escreve resultados em arquivo CSV
    FILE *f = fopen(out, "w");
    if (!f) { perror("fopen"); return 1; }
    
    // Header do CSV
    fprintf(f, "id,x,y,z,vx,vy,vz\n");
    
    // Escreve dados de cada partícula
    for (int i = 0; i < N; ++i) {
        fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", i,
                h_p[i].pos.x, h_p[i].pos.y, h_p[i].pos.z,
                h_p[i].vel.x, h_p[i].vel.y, h_p[i].vel.z);
    }
    fclose(f);

    printf("Wrote %s (N=%d, steps=%d)\n", out, N, steps);

    // Libera memória do device e host
    cudaFree(d_p);
    free(h_p);
    return 0;
}
