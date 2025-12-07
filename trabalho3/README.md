# Instalar as bibliotecas
```
mpicc bicho.c -o jogo_bicho -fopenmp
```

```
mpirun -np 4 ./jogo_bicho
```

```
export OMP_NUM_THREADS=2 && mpirun -np 2 ./jogo_bicho
```