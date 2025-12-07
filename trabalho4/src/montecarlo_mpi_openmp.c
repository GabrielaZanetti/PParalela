#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/*
    Simulação Monte Carlo - Jogo do Bicho (MPI + OpenMP)

    Este código:
    - executa simulações paralelas com MPI distribuindo o número de trials
    - usa OpenMP dentro de cada processo para paralelizar os sorteios
    - salva resultados em CSV para análise posterior
*/

int main(int argc, char *argv[]) {
    long long trials = 1000000;
    int chosen_animal = 5;
    double bet = 1.0;
    double payout = 20.0;
    unsigned int seed = 42;

    // leitura de parâmetros
    for(int i=1; i<argc; i++){
        if(strcmp(argv[i],"--trials")==0) trials = atoll(argv[++i]);
        else if(strcmp(argv[i],"--animal")==0) chosen_animal = atoi(argv[++i]);
        else if(strcmp(argv[i],"--bet")==0) bet = atof(argv[++i]);
        else if(strcmp(argv[i],"--payout")==0) payout = atof(argv[++i]);
        else if(strcmp(argv[i],"--seed")==0) seed = atoi(argv[++i]);
    }

    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long long local_trials = trials / world_size;
    unsigned int local_seed = seed + world_rank;

    double start_time = MPI_Wtime();

    long long local_wins = 0;
    double local_sum = 0.0;

    double local_compute_time_start = MPI_Wtime();

    #pragma omp parallel reduction(+:local_wins, local_sum)
    {
        unsigned int s = local_seed + omp_get_thread_num();
        #pragma omp for schedule(static)
        for(long long i = 0; i < local_trials; i++){
            int result = rand_r(&s) % 25 + 1;
            if(result == chosen_animal) {
                local_wins++;
                local_sum += (payout * bet - bet);
            } else {
                local_sum += (-bet);
            }
        }
    }

    double local_compute_time_end = MPI_Wtime();
    double local_compute_time = local_compute_time_end - local_compute_time_start;

    long long total_wins;
    double total_sum;

    MPI_Reduce(&local_wins, &total_wins, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    // salva CSV por processo
    char filename[100];
    sprintf(filename, "saida_rank_%d.csv", world_rank);

    FILE *fp = fopen(filename, "w");
    fprintf(fp,"rank,np,threads,trials_local,time_local,time_total,seed,wins_local,avg_return_local\n");
    fprintf(fp,"%d,%d,%d,%lld,%.6f,%.6f,%u,%lld,%.6f\n",
        world_rank, world_size, omp_get_max_threads(),
        local_trials, local_compute_time, total_time,
        local_seed, local_wins, local_sum/local_trials
    );
    fclose(fp);

    if(world_rank == 0) {
        printf("\n=== Simulação Monte Carlo MPI+OpenMP - Jogo do Bicho ===\n");
        printf("Processos MPI: %d | Threads por processo: %d\n", world_size, omp_get_max_threads());
        printf("Trials totais: %lld | Animal escolhido: %d | Payout: %.2f | Aposta: %.2f\n\n",
                trials, chosen_animal, payout, bet);

        double prob = (double)total_wins/trials;
        double media = total_sum/trials;

        printf("Vitórias totais: %lld\n", total_wins);
        printf("Probabilidade estimada: %.6f (teórica = 1/25 = 0.04)\n", prob);
        printf("Retorno médio por aposta: %.6f\n", media);
        printf("Tempo total execução (MPI+OpenMP): %.6f s\n", total_time);
        printf("CSV gerados: saida_rank_*.csv\n\n");
    }

    MPI_Finalize();
    return 0;
}
