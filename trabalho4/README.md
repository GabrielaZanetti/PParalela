# Trabalho 4 — Paralelismo Híbrido (MPI + OpenMP)
## Simulação Monte Carlo — Jogo do Bicho

Este repositório contém uma implementação híbrida (MPI + OpenMP) de uma simulação Monte Carlo que modela apostas no "Jogo do Bicho". O objetivo é estimar empiricamente a probabilidade de sucesso e o retorno esperado por aposta, bem como medir desempenho.

### Arquivos
- `montecarlo_mpi_openmp.c` — código fonte (C, MPI + OpenMP).
- `Makefile` — para compilar com `mpicc`.
- `run_example.sh` — script de exemplo para executar.
- `README.md` — este arquivo.

### Requisitos (WSL)
- WSL com distribuição Linux (Ubuntu recomendado).
- MPI (Open MPI ou MPICH) instalado. Exemplo: `sudo apt install openmpi-bin libopenmpi-dev`
- Compilador C com suporte OpenMP (o `mpicc` costuma invocar `gcc` com OpenMP).
- Recursos: ajuste `OMP_NUM_THREADS` e número de processos `-np` conforme CPU disponível.

### Compilar
No diretório que contém `montecarlo_mpi_openmp.c` e `Makefile`:
```bash
make
````

### Executar (exemplo)

Ajuste `NPROCS` e `OMP_NUM_THREADS` conforme sua máquina:

compilar
```bash
mpicc -fopenmp montecarlo_mpi_openmp.c -o montecarlo_mpi_openmp -lm
```

Execute múltiplos testes para alimentar o CSV
```bash
export OMP_NUM_THREADS=4

mpirun -np 1 ./montecarlo_mpi_openmp --trials 20000000
mpirun -np 2 ./montecarlo_mpi_openmp --trials 20000000
mpirun --oversubscribe -np 4 ./montecarlo_mpi_openmp --trials 20000000
mpirun --oversubscribe -np 8 ./montecarlo_mpi_openmp --trials 20000000
```

### Interpretação dos resultados

* `Vitórias` e `Probabilidade empírica` mostram quantas vezes o animal escolhido foi sorteado.
* `Retorno médio por trial (R$)` é o lucro líquido médio por aposta.
* Se o `Retorno médio` for negativo, a aposta é desfavorável.
* `Intervalo de confiança 95%` provê uma faixa para o retorno médio estimado.

### Observações

* Ajuste `payout_multiplier` para modelar diferentes regras do jogo.
* Experimentos de *strong scaling* e *weak scaling* podem ser realizados variando `-np` e `--trials`.
