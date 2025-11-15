// particles.cu
// Simple particle simulation on GPU. Particles attracted to origin.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

struct Particle { float3 pos; float3 vel; };

__global__ void integrate(Particle *p, int n, float dt, float damping, float G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 pos = p[i].pos;
    float3 vel = p[i].vel;

    // acceleration towards origin: a = -G * pos / (r^2 + eps)
    float r2 = pos.x*pos.x + pos.y*pos.y + pos.z*pos.z + 1e-6f;
    float invr = rsqrtf(r2);
    float invr3 = invr * invr * invr;
    float3 acc;
    acc.x = -G * pos.x * invr3;
    acc.y = -G * pos.y * invr3;
    acc.z = -G * pos.z * invr3;

    vel.x += acc.x * dt;
    vel.y += acc.y * dt;
    vel.z += acc.z * dt;

    // integrate
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    // damping
    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;

    p[i].pos = pos;
    p[i].vel = vel;
}

int main(int argc, char **argv) {
    int N = 20000;
    int steps = 200;
    float dt = 0.01f;
    const char *out = "particles.csv";

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) steps = atoi(argv[2]);
    if (argc >= 4) dt = atof(argv[3]);
    if (argc >= 5) out = argv[4];

    Particle *h_p = (Particle*)malloc(sizeof(Particle) * N);
    if (!h_p) { fprintf(stderr, "host alloc failed\n"); return 1; }

    // init random positions in sphere
    srand(1234);
    for (int i = 0; i < N; ++i) {
        float u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float w = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_p[i].pos = make_float3(u, v, w);
        h_p[i].vel = make_float3(0.0f, 0.0f, 0.0f);
    }

    Particle *d_p;
    checkCuda(cudaMalloc(&d_p, sizeof(Particle) * N), "cudaMalloc d_p");
    checkCuda(cudaMemcpy(d_p, h_p, sizeof(Particle) * N, cudaMemcpyHostToDevice), "cudaMemcpy h->d");

    int block = 256;
    int grid = (N + block - 1) / block;

    float damping = 0.999f;
    float G = 1.0f; // strength

    for (int s = 0; s < steps; ++s) {
        integrate<<<grid, block>>>(d_p, N, dt, damping, G);
        checkCuda(cudaGetLastError(), "kernel launch integrate");
    }

    checkCuda(cudaMemcpy(h_p, d_p, sizeof(Particle) * N, cudaMemcpyDeviceToHost), "cudaMemcpy d->h");

    FILE *f = fopen(out, "w");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "id,x,y,z,vx,vy,vz\n");
    for (int i = 0; i < N; ++i) {
        fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", i,
                h_p[i].pos.x, h_p[i].pos.y, h_p[i].pos.z,
                h_p[i].vel.x, h_p[i].vel.y, h_p[i].vel.z);
    }
    fclose(f);

    printf("Wrote %s (N=%d, steps=%d)\n", out, N, steps);

    cudaFree(d_p);
    free(h_p);
    return 0;
}
