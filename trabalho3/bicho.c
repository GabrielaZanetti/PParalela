#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

// Configurações do Jogo
#define NUM_BICHOS 25
#define MEU_PALPITE 0 // 0 = Avestruz

int main(int argc, char** argv) {
    int rank, size;
    long long total_apostas_global = 100000000; // 100 Milhões de apostas
    long long vitorias_locais = 0;
    long long vitorias_globais = 0;

    // Inicialização do MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide o trabalho pelos processos
    long long apostas_por_processo = total_apostas_global / size;

    double inicio = MPI_Wtime();

    // Início da Região Híbrida (OpenMP)
    // A variável vitorias_locais é somada de forma segura (reduction)
    #pragma omp parallel reduction(+:vitorias_locais)
    {
        // Semente única para garantir aleatoriedade real em paralelo
        unsigned int seed = time(NULL) + rank + omp_get_thread_num(); 
        
        int n_threads = omp_get_num_threads();
        // Divisão manual do trabalho entre threads deste processo
        long long apostas_por_thread = apostas_por_processo / n_threads;

        for (long long i = 0; i < apostas_por_thread; i++) {
            int sorteio = rand_r(&seed) % 10000; // 0000 a 9999
            int dezena = sorteio % 100;
            
            int bicho_sorteado;
            if (dezena == 0) {
                bicho_sorteado = 24; // Vaca
            } else {
                bicho_sorteado = (dezena - 1) / 4;
            }

            if (bicho_sorteado == MEU_PALPITE) {
                vitorias_locais++;
            }
        }
    }

    // MPI Reduction: Junta tudo no processo Mestre (rank 0)
    MPI_Reduce(&vitorias_locais, &vitorias_globais, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double fim = MPI_Wtime();

    if (rank == 0) {
        printf("=== JOGO DO BICHO: MPI + OPENMP (Monte Carlo) ===\n");
        printf("Processos MPI: %d\n", size);
        #pragma omp parallel 
        {
            #pragma omp single
            printf("Threads OpenMP por Processo: %d\n", omp_get_num_threads());
        }
        printf("Total de Apostas: %lld\n", total_apostas_global);
        printf("Vitorias do Avestruz: %lld\n", vitorias_globais);
        
        double probabilidade = (double)vitorias_globais / total_apostas_global * 100.0;
        printf("Probabilidade: %.4f%% (Esperado ~4.00%%)\n", probabilidade);
        printf("Tempo: %f segundos\n", fim - inicio);
    }

    MPI_Finalize();
    return 0;
}