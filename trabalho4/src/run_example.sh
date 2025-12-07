#!/bin/bash
# run_example.sh
# Exemplo de execução (ajuste número de processos, trials, seed conforme necessário)

NPROCS=4
TRIALS=20000000        # 20 milhões (ajuste se necessário)
BET=1.0
ANIMAL=7
PAYOUT=20.0
SEED=123456

# caso queira ajustar OMP_NUM_THREADS:
export OMP_NUM_THREADS=4

mpirun -np ${NPROCS} ./montecarlo_mpi_openmp --trials ${TRIALS} --bet ${BET} --animal ${ANIMAL} --payout ${PAYOUT} --seed ${SEED}
