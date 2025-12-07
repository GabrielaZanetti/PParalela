# Makefile para compilar a versão MPI + OpenMP
CC = mpicc
CFLAGS = -O3 -fopenmp -std=c99
TARGET = montecarlo_mpi_openmp
SRC = montecarlo_mpi_openmp.c

all: build

build:
\t$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
\trm -f $(TARGET) *.o

.PHONY: all build clean
