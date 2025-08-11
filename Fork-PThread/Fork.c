#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

#define ROWS 4
#define COLS 4

int main() {
    int matrix[ROWS][COLS] = { {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16} };

    int fd[2];
    if (pipe(fd) == -1) {
        perror("pipe");
        exit(1);
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(1);
    }

    if (pid == 0) {
        close(fd[0]);
        int sum_child = 0;
        for (int i = ROWS / 2; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                sum_child += matrix[i][j];
            }
        }
        write(fd[1], &sum_child, sizeof(sum_child));
        close(fd[1]);
        exit(0);
    } else {
        close(fd[1]);
        int sum_parent = 0;
        for (int i = 0; i < ROWS / 2; i++) {
            for (int j = 0; j < COLS; j++) {
                sum_parent += matrix[i][j];
            }
        }
        int sum_child;
        read(fd[0], &sum_child, sizeof(sum_child));
        close(fd[0]);
        wait(NULL);
        printf("Somatório total (fork): %d\n", sum_parent + sum_child);
    }

    return 0;
}
