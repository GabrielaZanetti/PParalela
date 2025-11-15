# Makefile for cuda_examples

NVCC ?= nvcc
CFLAGS ?= -O3

all: mandelbrot particles

mandelbrot: mandelbrot.cu
	$(NVCC) $(CFLAGS) mandelbrot.cu -o mandelbrot

particles: particles.cu
	$(NVCC) $(CFLAGS) particles.cu -o particles

clean:
	rm -f mandelbrot particles *.ppm *.csv

.PHONY: all clean
