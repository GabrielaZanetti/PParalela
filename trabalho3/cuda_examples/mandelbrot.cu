// mandelbrot.cu
// Simple CUDA Mandelbrot renderer -> PPM (P6)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void mandelbrot_kernel(unsigned char *img, int w, int h, int maxIter,
                                  double xmin, double xmax, double ymin, double ymax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    double cx = xmin + (double)x * (xmax - xmin) / (double)(w - 1);
    double cy = ymin + (double)y * (ymax - ymin) / (double)(h - 1);

    double zx = 0.0, zy = 0.0;
    int iter = 0;
    while (zx*zx + zy*zy <= 4.0 && iter < maxIter) {
        double xt = zx*zx - zy*zy + cx;
        zy = 2.0*zx*zy + cy;
        zx = xt;
        ++iter;
    }

    int idx = 3 * (y * w + x);
    if (iter == maxIter) {
        img[idx+0] = img[idx+1] = img[idx+2] = 0; // black
    } else {
        // simple coloring: smooth grayscale with gamma
        double t = (double)iter / (double)maxIter;
        unsigned char v = (unsigned char)(255.0 * sqrt(t));
        // tint for nicer output
        img[idx+0] = (unsigned char) (v * 0.6); // R
        img[idx+1] = (unsigned char) (v * 0.9); // G
        img[idx+2] = v;                         // B
    }
}

int main(int argc, char **argv) {
    int w = 1024, h = 768, maxIter = 1000;
    const char *out = "mandelbrot.ppm";
    double xmin = -2.0, xmax = 1.0, ymin = -1.2, ymax = 1.2;

    if (argc >= 3) { w = atoi(argv[1]); h = atoi(argv[2]); }
    if (argc >= 4) maxIter = atoi(argv[3]);
    if (argc >= 5) out = argv[4];

    size_t imgSize = (size_t)w * (size_t)h * 3;
    unsigned char *h_img = (unsigned char*)malloc(imgSize);
    if (!h_img) { fprintf(stderr, "Allocation failed\n"); return 1; }

    unsigned char *d_img;
    checkCuda(cudaMalloc(&d_img, imgSize), "cudaMalloc d_img");

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    mandelbrot_kernel<<<grid, block>>>(d_img, w, h, maxIter, xmin, xmax, ymin, ymax);
    checkCuda(cudaGetLastError(), "Kernel launch failed");
    checkCuda(cudaMemcpy(h_img, d_img, imgSize, cudaMemcpyDeviceToHost), "cudaMemcpy d->h");

    // write PPM P6
    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(h_img, 1, imgSize, f);
    fclose(f);

    printf("Wrote %s (%dx%d)\n", out, w, h);

    cudaFree(d_img);
    free(h_img);
    return 0;
}
