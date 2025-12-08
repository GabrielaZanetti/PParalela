// montecarlo_mpi_openmp.c
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/*
    Simulação Monte Carlo - Jogo do Bicho (MPI + OpenMP)

    - MPI divide o total de simulações entre processos
    - OpenMP paraleliza os sorteios internamente
    - Resultados globais gravados em CSV para gráficos de desempenho
*/

int main(int argc, char *argv[]) {

    long long trials = 1000000;
    int chosen_animal = 5;
    double bet = 1.0, payout = 20.0;
    unsigned int seed = 42;

    // leitura de parâmetros
    for(int i=1; i<argc; i++){
        if(strcmp(argv[i],"--trials")==0) trials = atoll(argv[++i]);
        else if(strcmp(argv[i],"--animal")==0) chosen_animal = atoi(argv[++i]);
        else if(strcmp(argv[i],"--bet")==0) bet = atof(argv[++i]);
        else if(strcmp(argv[i],"--payout")==0) payout = atof(argv[++i]);
        else if(strcmp(argv[i],"--seed")==0) seed = atoi(argv[++i]);
    }

    MPI_Init(&argc,&argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    long long local_trials = trials / size;
    unsigned int local_seed = seed + rank;

    double time_start = MPI_Wtime();

    long long local_wins = 0;
    double local_sum = 0.0;

    double t0 = MPI_Wtime();
    #pragma omp parallel reduction(+:local_wins, local_sum)
    {
        unsigned int s = local_seed + omp_get_thread_num();
        #pragma omp for schedule(static)
        for(long long i=0;i<local_trials;i++){
            int result = rand_r(&s)%25 + 1;
            if(result == chosen_animal){
                local_wins++;
                local_sum += (payout*bet - bet);
            } else local_sum -= bet;
        }
    }
    double t1 = MPI_Wtime();

    long long wins_total;
    double sum_total;

    MPI_Reduce(&local_wins,&wins_total,1,MPI_LONG_LONG_INT,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&local_sum,&sum_total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    double time_end = MPI_Wtime();
    double total_time = time_end - time_start;

    // CSV individual por processo
    char fname[50];
    sprintf(fname,"saida_rank_%d.csv",rank);
    FILE *f = fopen(fname,"w");
    fprintf(f,"rank,np,threads,trials_local,time_local,time_total,seed,wins_local,avg_return_local\n");
    fprintf(f,"%d,%d,%d,%lld,%.6f,%.6f,%u,%lld,%.6f\n",
        rank,size,omp_get_max_threads(),local_trials,(t1-t0),total_time,local_seed,local_wins,local_sum/local_trials);
    fclose(f);

    // ================= NOVA PARTE: REGISTRO GLOBAL PARA GRÁFICO =================
    if(rank == 0){
        FILE *fr = fopen("resultados_execucao.csv","a"); // acumula runs
        fprintf(fr,"%d,%d,%lld,%.6f\n",size,omp_get_max_threads(),trials,total_time);
        fclose(fr);

        printf("\n=== Simulação Monte Carlo MPI+OpenMP - Jogo do Bicho ===\n");
        printf("Processos MPI: %d | Threads: %d\n", size, omp_get_max_threads());
        printf("Trials totais: %lld | Animal: %d | Payout: %.2f | Aposta: %.2f\n\n",
                trials,chosen_animal,payout,bet);

        printf("Vitórias totais: %lld\n",wins_total);
        printf("Probabilidade estimada: %.6f\n",(double)wins_total/trials);
        printf("Retorno médio por aposta: %.6f\n",sum_total/trials);
        printf("Tempo total execução: %.6f s\n",total_time);
        printf("Dados adicionados em: resultados_execucao.csv\n\n");
    }

    MPI_Finalize();
    return 0;
}
