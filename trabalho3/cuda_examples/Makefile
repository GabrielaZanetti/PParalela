# Makefile for CUDA Mandelbrot Example

NVCC ?= nvcc
CFLAGS ?= -O3

all: mandelbrot

mandelbrot: mandelbrot.cu
	$(NVCC) $(CFLAGS) mandelbrot.cu -o mandelbrot

clean:
	rm -f mandelbrot *.ppm

.PHONY: all clean
