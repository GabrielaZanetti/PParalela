#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define MAX_LINHA 5000
#define MAX_PALAVRA 128
#define TAM_HASH 80000
#define PORTA_HTTP 8080


//nó de lista encadeada pra resolver as colisões na tabela hash
typedef struct No {
    char palavra[MAX_PALAVRA];
    long contagem;
    struct No *prox;
} No;

typedef struct {
    No *tabela[TAM_HASH];
} HashPalavras;

pthread_t server_thread; //controle do servidor HTTP
volatile sig_atomic_t server_running = 1;
int server_sock = -1;

//função djb2 para gerar o índice de bucket - da uma boa distribuição e custo baixo
unsigned long hash_fn(const char *str) {
    unsigned long h = 5381;
    int c;
    while ((c = (unsigned char)*str++)) h = ((h << 5) + h) + c;
    return h % TAM_HASH;
}

void inicializar_hash(HashPalavras *h) { //zera os buckets
    for (int i = 0; i < TAM_HASH; i++) h->tabela[i] = NULL;
}

//procurar a palavra no bucket do hash se existir somar, se não existir cria com um valor inicial
void inserir_com_contagem(HashPalavras *h, const char *palavra, long cont) {
    unsigned long idx = hash_fn(palavra);
    No *at = h->tabela[idx];
    while (at) {
        if (strcmp(at->palavra, palavra) == 0) { at->contagem += cont; return; }
        at = at->prox;
    }
    No *novo = (No*)malloc(sizeof(No));
    if (!novo) return;
    strncpy(novo->palavra, palavra, MAX_PALAVRA-1);
    novo->palavra[MAX_PALAVRA-1] = '\0';
    novo->contagem = cont;
    novo->prox = h->tabela[idx];
    h->tabela[idx] = novo;
}
void inserir(HashPalavras *h, const char *palavra) {
    inserir_com_contagem(h, palavra, 1);
}

//apenas para evitar vazamento de memória, só para desalocar os nós das lista
void liberar_hash(HashPalavras *h) {
    for (int i = 0; i < TAM_HASH; i++) {
        No *at = h->tabela[i];
        while (at) { No *tmp = at; at = at->prox; free(tmp); }
        h->tabela[i] = NULL;
    }
}

//limpeza de letras, converte para minúsculas
void normalizar(char *palavra) {
    int i=0,j=0;
    for (; palavra[i]; i++) {
        unsigned char c = (unsigned char)palavra[i];
        if (isalpha(c) || c == '\'') palavra[j++] = (char)tolower(c);
    }
    palavra[j] = '\0';
}

//extrai apenas as letras da 4ª coluna do CSV
//também consideração de vírgulas dentro de aspas
int separar_csv(const char *linha, char *letra_out) {
    int aspas = 0, campo = 0, i = 0;
    const char *start = NULL;
    for (; linha[i]; i++) {
        char c = linha[i];
        if (c == '"') aspas = !aspas;
        else if (c == ',' && !aspas) {
            campo++;
            if (campo == 3) { start = &linha[i + 1]; break; }
        }
    }
    if (!start) return -1;
    while (*start && isspace((unsigned char)*start)) start++;
    if (*start == '"') start++;
    strncpy(letra_out, start, MAX_LINHA - 1);
    letra_out[MAX_LINHA - 1] = '\0';
    size_t len = strlen(letra_out);
    if (len > 0) {
        char *end = letra_out + len - 1;
        while (end >= letra_out && isspace((unsigned char)*end)) { *end = '\0'; end--; }
        if (end >= letra_out && *end == '"') *end = '\0';
    }
    return 0;
}

//map da contagem de palavras
void processar_letra(HashPalavras *h, const char *letra) {
    //tokeniza a letra em palavras normaliza e insere no hash
    char token[MAX_PALAVRA];
    int i = 0, j = 0;
    for (;;) {
        unsigned char c = (unsigned char)letra[i];
        if (isalpha(c) || c == '\'') { if (j < MAX_PALAVRA - 1) token[j++] = (char)c; }
        else {
            if (j > 0) {
                token[j] = '\0';
                normalizar(token);
                if (token[0] != '\0') inserir(h, token);
                j = 0;
            }
        }
        if (c == '\0') break;
        i++;
    }
}

//comunicação MPI para redução centralizada
// ele envia o processo - palavra, contagem -
void enviar_hash(HashPalavras *h, int destino) {
    int total = 0;
    for (int i = 0; i < TAM_HASH; i++)
        for (No *n = h->tabela[i]; n; n = n->prox) total++;
    //envia quantos pares virão
    MPI_Send(&total, 1, MPI_INT, destino, 0, MPI_COMM_WORLD);
    for (int i = 0; i < TAM_HASH; i++) {
        //pra cada par ele envia o tamanho da string, string e contagem
        for (No *n = h->tabela[i]; n; n = n->prox) {
            int tam = (int)strlen(n->palavra) + 1;
            MPI_Send(&tam, 1, MPI_INT, destino, 0, MPI_COMM_WORLD);
            MPI_Send(n->palavra, tam, MPI_CHAR, destino, 0, MPI_COMM_WORLD);
            MPI_Send(&n->contagem, 1, MPI_LONG, destino, 0, MPI_COMM_WORLD);
        }
    }
}

void receber_hash(HashPalavras *h, int origem) {
    int total;
    MPI_Recv(&total, 1, MPI_INT, origem, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < total; i++) {
        int tam;
        char palavra[MAX_PALAVRA];
        long cont;
        MPI_Recv(&tam, 1, MPI_INT, origem, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (tam > MAX_PALAVRA) tam = MAX_PALAVRA;
        MPI_Recv(palavra, tam, MPI_CHAR, origem, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        palavra[tam-1] = '\0';
        MPI_Recv(&cont, 1, MPI_LONG, origem, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (cont > 0) inserir_com_contagem(h, palavra, cont);
    }
}

//saída do http
typedef struct { char palavra[MAX_PALAVRA]; long cont; } Par;
//qsort para contagem decrescente
int par_cmp(const void *a, const void *b) {
    const Par *pa = (const Par*)a;
    const Par *pb = (const Par*)b;
    if (pa->cont < pb->cont) return 1;
    if (pa->cont > pb->cont) return -1;
    return strcmp(pa->palavra, pb->palavra);
}
//extrai as entradas do hash ordena e formata
char *gerar_resposta_texto(HashPalavras *h) {
    size_t n = 0;
    for (int i = 0; i < TAM_HASH; i++)
        for (No *no = h->tabela[i]; no; no = no->prox) n++;
    if (n == 0) return strdup("Nenhuma palavra contada.\n");
    Par *arr = (Par*)malloc(sizeof(Par) * n);
    if (!arr) return strdup("Erro de memória.\n");
    size_t idx = 0;
    for (int i = 0; i < TAM_HASH; i++) {
        for (No *no = h->tabela[i]; no; no = no->prox) {
            strncpy(arr[idx].palavra, no->palavra, MAX_PALAVRA-1);
            arr[idx].palavra[MAX_PALAVRA-1] = '\0';
            arr[idx].cont = no->contagem;
            idx++;
        }
    }
    qsort(arr, n, sizeof(Par), par_cmp);
    size_t bufsize = 0;
    for (size_t i = 0; i < n; i++) bufsize += strlen(arr[i].palavra) + 32;
    char *buf = (char*)malloc(bufsize + 1);
    if (!buf) { free(arr); return strdup("Erro de memória.\n"); }
    buf[0] = '\0';
    char tmp[256];
    for (size_t i = 0; i < n; i++) {
        snprintf(tmp, sizeof(tmp), "%s: %ld\n", arr[i].palavra, arr[i].cont);
        strcat(buf, tmp);
    }
    free(arr);
    return buf;
}

void handle_sigint(int sig) {
    (void)sig;
    server_running = 0;
    if (server_sock >= 0) close(server_sock);
}
//servidor HTTP simples apenas para a vizualição
void *http_server(void *arg) {
    HashPalavras *h = (HashPalavras*)arg;
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
    printf("Servidor HTTP em http://localhost:%d/ (Ctrl+C para sair)\n", PORTA_HTTP);
    while (server_running) {
        struct sockaddr_in client;
        socklen_t client_len = sizeof(client);
        int client_sock = accept(server_sock, (struct sockaddr*)&client, &client_len);
        if (client_sock < 0) {
            if (!server_running) break;
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        char req[1024];
        ssize_t r = recv(client_sock, req, sizeof(req)-1, 0);
        if (r <= 0) { close(client_sock); continue; }
        req[r] = '\0';
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
    if (server_sock >= 0) { close(server_sock); server_sock = -1; }
    return NULL;
}

//oquestra MPI, leitura do CSV e a distribuição do trabalho
int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "mpirun -np 4 contagem_mpi spotify_cleaned.csv ");
        MPI_Finalize();
        return 1;
    }

    FILE *f = fopen(argv[1], "r");

    //hash onde cada rank acumula suas contagens
    HashPalavras local_hash;
    inicializar_hash(&local_hash);

    char linha[MAX_LINHA];
    int linha_id = 0;

    //pula cabeçalho do csv
    if (!fgets(linha, MAX_LINHA, f)) {
        fclose(f);
        MPI_Finalize();
        return 1;
    }

    //distribuição simples por linha (id % size) == rank
    while (fgets(linha, MAX_LINHA, f)) {
        if ((linha_id % size) == rank) {
            char letra[MAX_LINHA];
            if (separar_csv(linha, letra) == 0) processar_letra(&local_hash, letra);
        }
        linha_id++;
    }
    fclose(f);

    if (rank == 0) {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = handle_sigint;
        sigaction(SIGINT, &sa, NULL);

        for (int i = 1; i < size; i++) receber_hash(&local_hash, i);

        if (pthread_create(&server_thread, NULL, http_server, &local_hash) != 0) {
            fprintf(stderr, "Falha ao criar thread do servidor HTTP\n");
        } else {
            pthread_join(server_thread, NULL);
        }
    } else {
        enviar_hash(&local_hash, 0);
    }

    liberar_hash(&local_hash);
    MPI_Finalize();
    return 0;
}
