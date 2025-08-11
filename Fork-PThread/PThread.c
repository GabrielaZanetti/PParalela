#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define ROWS 4
#define COLS 4

int matrix[ROWS][COLS] = { {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16} };

typedef struct {
    int start_row;
    int end_row;
    int sum;
} ThreadData;

void* sum_matrix(void* arg) {
    ThreadData* data = (ThreadData*) arg;
    data->sum = 0;
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < COLS; j++) {
            data->sum += matrix[i][j];
        }
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;
    ThreadData d1 = {0, ROWS / 2, 0};
    ThreadData d2 = {ROWS / 2, ROWS, 0};

    pthread_create(&t1, NULL, sum_matrix, &d1);
    pthread_create(&t2, NULL, sum_matrix, &d2);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    int total = d1.sum + d2.sum;
    printf("Somatório total (pthread): %d\n", total);

    return 0;
}
