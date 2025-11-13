#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define MAX_LINHA 8192
#define MAX_NOME_ARTISTA 256
#define TAM_HASH 70000
#define PORTA_HTTP 8081

// nó da hash lista pra colisão
typedef struct No {
    char item[MAX_NOME_ARTISTA];
    long contagem;
    struct No *prox;
} No;

typedef struct {
    No *tabela[TAM_HASH];
} HashContagem;

pthread_t server_thread;
int server_running = 1;
int server_sock = -1;

// hash djb2 simples
unsigned long hash_fn(const char *str) {
    unsigned long h = 5381;
    int c;
    while ((c = (unsigned char)*str++)) h = ((h << 5) + h) + c;
    return h % TAM_HASH;
}

// zera buckets
void inicializar_hash(HashContagem *h) {
    for (int i = 0; i < TAM_HASH; i++) h->tabela[i] = NULL;
}

// somar se existe, criar se não, genérica: serve pra +1 e pra merge
void inserir_com_contagem(HashContagem *h, const char *item, long cont) {
    unsigned long idx = hash_fn(item);
    No *at = h->tabela[idx];
    while (at) {
        if (strcmp(at->item, item) == 0) { at->contagem += cont; return; }
        at = at->prox;
    }
    No *novo = (No*)malloc(sizeof(No));
    if (!novo) return;
    strncpy(novo->item, item, MAX_NOME_ARTISTA-1);
    novo->item[MAX_NOME_ARTISTA-1] = '\0';
    novo->contagem = cont;
    novo->prox = h->tabela[idx];
    h->tabela[idx] = novo;
}

// desaloca nós (sem vazamento)
void liberar_hash(HashContagem *h) {
    for (int i = 0; i < TAM_HASH; i++) {
        No *at = h->tabela[i];
        while (at) { No *tmp = at; at = at->prox; free(tmp); }
        h->tabela[i] = NULL;
    }
}

int extrair_artista(const char *linha, char *artista_out) {
    // extrai artista do 1º campo CSV, lida com "" escapadas
    int i = 0, j = 0;
    while (linha[i] && isspace((unsigned char)linha[i])) i++;
    if (linha[i] != '"') return -1;
    i++;
    while (linha[i] && j < MAX_NOME_ARTISTA - 1) {
        if (linha[i] == '"') {
            if (linha[i+1] == '"') { artista_out[j++] = '"'; i += 2; }
            else break;
        } else {
            artista_out[j++] = linha[i++];
        }
    }
    artista_out[j] = '\0';
    if (linha[i] != '"') return -1;
    return 0;
}

void enviar_hash(HashContagem *h, int destino) { // envia pares (item, cont) pro destino
    int total = 0;
    for (int i = 0; i < TAM_HASH; i++)
        for (No *n = h->tabela[i]; n; n = n->prox) total++;

    MPI_Send(&total, 1, MPI_INT, destino, 0, MPI_COMM_WORLD);

    for (int i = 0; i < TAM_HASH; i++) {
        for (No *n = h->tabela[i]; n; n = n->prox) {
            int tam = (int)strlen(n->item) + 1;
            MPI_Send(&tam, 1, MPI_INT, destino, 0, MPI_COMM_WORLD);
            MPI_Send(n->item, tam, MPI_CHAR, destino, 0, MPI_COMM_WORLD);
            MPI_Send(&n->contagem, 1, MPI_LONG, destino, 0, MPI_COMM_WORLD);
        }
    }
}

void receber_hash(HashContagem *h, int origem) { // recebe pares do origem e agrega
    int total;
    MPI_Recv(&total, 1, MPI_INT, origem, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < total; i++) {
        int tam;
        char item[MAX_NOME_ARTISTA];
        long cont;
        MPI_Recv(&tam, 1, MPI_INT, origem, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (tam > MAX_NOME_ARTISTA) tam = MAX_NOME_ARTISTA;
        MPI_Recv(item, tam, MPI_CHAR, origem, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        item[tam-1] = '\0';
        MPI_Recv(&cont, 1, MPI_LONG, origem, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (cont > 0) inserir_com_contagem(h, item, cont);
    }
}

// par auxiliar pra ordenar
typedef struct { char item[MAX_NOME_ARTISTA]; long cont; } Par;

// ordena por cont decrescente
int par_cmp(const void *a, const void *b) {
    const Par *pa = (const Par*)a, *pb = (const Par*)b;
    if (pa->cont < pb->cont) return 1;
    if (pa->cont > pb->cont) return -1;
    return strcmp(pa->item, pb->item);
}

// monta texto da resposta HTTP
char *gerar_resposta_texto(HashContagem *h) {
    size_t n = 0;
    for (int i = 0; i < TAM_HASH; i++)
        for (No *no = h->tabela[i]; no; no = no->prox) n++;
    if (n == 0) return strdup("Nenhum artista contado.\n");

    Par *arr = (Par*)malloc(sizeof(Par) * n);
    if (!arr) return strdup("Erro de memoria.\n");

    size_t idx = 0;
    for (int i = 0; i < TAM_HASH; i++) {
        for (No *no = h->tabela[i]; no; no = no->prox) {
            strncpy(arr[idx].item, no->item, MAX_NOME_ARTISTA-1);
            arr[idx].item[MAX_NOME_ARTISTA-1] = '\0';
            arr[idx].cont = no->contagem;
            idx++;
        }
    }

    qsort(arr, n, sizeof(Par), par_cmp);

    size_t bufsize = 0;
    for (size_t i = 0; i < n; i++) bufsize += strlen(arr[i].item) + 32;

    char *buf = (char*)malloc(bufsize + 1);
    if (!buf) { free(arr); return strdup("Erro de memoria.\n"); }
    buf[0] = '\0';

    char tmp[MAX_NOME_ARTISTA + 50];
    for (size_t i = 0; i < n; i++) {
        snprintf(tmp, sizeof(tmp), "%s: %ld\n", arr[i].item, arr[i].cont);
        strcat(buf, tmp);
    }
    free(arr);
    return buf;
}

// servidor HTTP simples (só pra ver o resultado)
void *http_server(void *arg) {
    HashContagem *h = (HashContagem*)arg;

    struct sockaddr_in addr;
    server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) { perror("socket"); return NULL; }

    int opt = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(PORTA_HTTP);

    if (bind(server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); close(server_sock); return NULL; }
    if (listen(server_sock, 5) < 0) { perror("listen"); close(server_sock); return NULL; }

    printf("HTTP em http://localhost:%d/\n", PORTA_HTTP);

    while (server_running) {
        int client_sock = accept(server_sock, NULL, NULL);
        if (client_sock < 0) { if (errno == EINTR) continue; break; }

        char req[1024];
        ssize_t r = recv(client_sock, req, sizeof(req)-1, 0);
        if (r <= 0) { close(client_sock); continue; }
        req[r] = '\0'; // garante termino

        char *body = gerar_resposta_texto(h);
        if (!body) body = strdup("Erro interno\n");

        char header[256];
        snprintf(header, sizeof(header),
                 "HTTP/1.1 200 OK\r\n"
                 "Content-Type: text/plain; charset=utf-8\r\n"
                 "Content-Length: %zu\r\n"
                 "Connection: close\r\n\r\n",
                 strlen(body));
        send(client_sock, header, strlen(header), 0);
        send(client_sock, body, strlen(body), 0);
        free(body);
        close(client_sock);
    }

    if (server_sock >= 0) close(server_sock);
    return NULL;
}

// orquestra MPI + leitura e contagem por linha
int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "mpirun -np 4 artistas_mpi spotify_cleaned.csv");
        MPI_Finalize();
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    HashContagem local_hash;
    inicializar_hash(&local_hash);

    char linha[MAX_LINHA];
    int linha_id = 0;

    // pula cabeçalho
    if (!fgets(linha, MAX_LINHA, f)) { fclose(f); MPI_Finalize(); return 1; }

    // partição simples: (linha_id % size) == rank
    while (fgets(linha, MAX_LINHA, f)) {
        if ((linha_id % size) == rank) {
            char artista[MAX_NOME_ARTISTA];
            if (extrair_artista(linha, artista) == 0 && artista[0] != '\0') {
                inserir_com_contagem(&local_hash, artista, 1); // contagem normal (+1)
            }
        }
        linha_id++;
    }
    fclose(f);

    if (rank == 0) {
        for (int i = 1; i < size; i++) receber_hash(&local_hash, i);
        if (pthread_create(&server_thread, NULL, http_server, &local_hash) == 0) {
            pthread_join(server_thread, NULL);
        } else {
            fprintf(stderr, "falha ao criar servidor HTTP");
        }
    } else {
        enviar_hash(&local_hash, 0);
    }

    liberar_hash(&local_hash);
    MPI_Finalize();
    return 0;
}
