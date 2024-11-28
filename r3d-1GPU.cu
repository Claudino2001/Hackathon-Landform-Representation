#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>  // Para CUDA Runtime

#define Mxx 5026
#define Mxy 5026
#define INPUT "mde.pnz"
#define OUTPUT "output.txt"
#define ANOVA "anova.txt"

// Protótipos das funções
long int n = 0;  // Número global de pontos
double zm = 0.0;  // Média de z

void dados(double *x, double *y, double *z);
void matrizes(double *A, double *B, double *x, double *y, double *z, int N, int r);
void gauss_elimination(double *A, double *B, int N);
void anova(double *A, double *x, double *y, double *z, int N, int r);

__global__ void calculate_matrices_kernel(double *A, double *B, double *x, double *y, double *z, int N, int n, int r) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;  // Índice para linhas de A
    int c = blockIdx.y * blockDim.y + threadIdx.y;  // Índice para colunas de A

    if (l < N && c < N) {
        double local_A = 0.0;
        double local_B = 0.0;

        // Cálculo das matrizes A e B
        for (int i = 0; i < n; ++i) {
            int xIdx = (int)(l / (r + 1)) + (int)(c / (r + 1));
            int yIdx = l % (r + 1) + c % (r + 1);

            local_A += pow(x[i], xIdx) * pow(y[i], yIdx);
            if (c == 0) {
                local_B += z[i] * pow(x[i], xIdx) * pow(y[i], yIdx);
            }
        }

        A[l + c * N] = local_A;
        if (c == 0) {
            B[l] = local_B;
        }
    }
}

int main(int argc, char **argv) {
    int r = atoi(argv[1]);  // Grau do polinômio
    int s = r;               // Simplify Polynomial Degree
    int N = (r + 1) * (s + 1);  // Número de coeficientes de polinômio
    int MAX = (Mxx + 1) * (Mxy + 1);

    double *A = (double*)malloc(sizeof(double) * N * N);
    double *B = (double*)malloc(sizeof(double) * N);
    double *x = (double*)malloc(sizeof(double) * MAX);
    double *y = (double*)malloc(sizeof(double) * MAX);
    double *z = (double*)malloc(sizeof(double) * MAX);

    double t1, t2;

    // Inicializa o tempo
    t1 = omp_get_wtime();

    // Etapas do processo
    dados(x, y, z);

    // Aloca memória na GPU para A, B, x, y, z
    double *d_A, *d_B, *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_A, sizeof(double) * N * N);
    cudaMalloc((void **)&d_B, sizeof(double) * N);
    cudaMalloc((void **)&d_x, sizeof(double) * n);
    cudaMalloc((void **)&d_y, sizeof(double) * n);
    cudaMalloc((void **)&d_z, sizeof(double) * n);

    // Copia dados para a memória da GPU
    cudaMemcpy(d_x, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, sizeof(double) * n, cudaMemcpyHostToDevice);

    // Definir blocos e grids para o kernel
    dim3 blockDim(16, 16);  // Blocos de 16x16 threads
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Lançar o kernel para calcular A e B
    calculate_matrices_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_x, d_y, d_z, N, n, r);

    // Sincroniza a execução da GPU
    cudaDeviceSynchronize();

    // Copiar os resultados de volta para a CPU
    cudaMemcpy(A, d_A, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, sizeof(double) * N, cudaMemcpyDeviceToHost);

    // Libera a memória da GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    // Resolve o sistema linear utilizando eliminação de Gauss
    gauss_elimination(A, B, N);

    // Realiza a análise de variância (ANOVA)
    anova(B, x, y, z, N, r);

    // Finaliza o tempo
    t2 = omp_get_wtime();

    printf("%d\t%5.2f\n", r, t2 - t1);

    // Libera a memória alocada
    free(A);
    free(B);
    free(x);
    free(y);
    free(z);

    return 0;
}

void dados(double *x, double *y, double *z) {
    int col, row, count = -1;
    FILE *f;
    n = 0;  // Inicializa n aqui

    if ((f = fopen(INPUT, "r")) == NULL) {
        printf("\n Erro I/O\n");
        exit(1);
    }

    for (row = 0; row < Mxy; ++row) {
        for (col = 0; col < Mxx; ++col) {
            float h;
            int result = fscanf(f, "%f", &h);
            count++;
            if (count % 10 != 0 || h <= 0) continue;
            x[n] = row;
            y[n] = col;
            z[n] = h / 2863.0;
            n++;  // Incrementa n para o próximo ponto
        }
    }

    // Calcula a média de z para armazenar em zm
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += z[i];
    }
    zm = sum / n;  // Calcula a média

    fclose(f);
}

void gauss_elimination(double *A, double *B, int N) {
    for (int k = 0; k < N; ++k) {
        // Encontrar o pivô
        double pivot = A[k + k * N];
        if (pivot == 0.0) {
            printf("Pivô nulo encontrado, o sistema é singular.\n");
            exit(1);
        }

        // Normaliza a linha
        for (int j = k + 1; j < N; ++j) {
            A[k + j * N] /= pivot;
        }
        B[k] /= pivot;

        // Elimina os elementos abaixo do pivô
        for (int i = k + 1; i < N; ++i) {
            double factor = A[i + k * N];
            for (int j = k + 1; j < N; ++j) {
                A[i + j * N] -= factor * A[k + j * N];
            }
            B[i] -= factor * B[k];
        }
    }

    // Substituição para trás
    for (int k = N - 1; k >= 0; --k) {
        for (int i = k - 1; i >= 0; --i) {
            double factor = A[i + k * N];
            A[i + k * N] = 0.0;
            B[i] -= factor * B[k];
        }
    }
}

void anova(double *A, double *x, double *y, double *z, int N, int r) {
    int i, glReg, glR, glT, c, l;
    double SQReg, SQR, SQT, QMReg, QMR, R2, F, ze;
    FILE *f;
    int s = r;

    SQR = SQReg = 0.0;
    glReg = N;
    glR = n - 2 * N;
    glT = n - N;

    for (i = 0; i < n; ++i) {  // Usando n aqui
        ze = 0.0;
        for (c = 0; c < r + 1; ++c)
            for (l = 0; l < s + 1; ++l)
                ze += A[c + l * (r + 1)] * pow(x[i], c) * pow(y[i], l);

        SQReg += (ze - zm) * (ze - zm);
        SQR += (z[i] - ze) * (z[i] - ze);
    }

    SQT = SQReg + SQR;
    QMReg = SQReg / glReg;
    QMR = SQR / glR;
    F = QMReg / QMR;
    R2 = SQReg / SQT;

    if ((f = fopen(ANOVA, "w")) == NULL) {
        printf("\n Error I/O\n");
        exit(2);
    }

    fprintf(f, " \n\n\n\n");
    fprintf(f, " ANOVA\n");
    fprintf(f, " =================================================\n");
    fprintf(f, " FV           gl      SQ         QM          F    \n");
    fprintf(f, " =================================================\n");
    fprintf(f, " Regression  %5d  %12e  %12e  %12e\n", glReg, SQReg, QMReg, F);
    fprintf(f, " Residue     %5d  %12e  %12e      \n", glR, SQR, QMR);
    fprintf(f, " -------------------------------------------------\n");
    fprintf(f, " Total      %5d  %12e            \n", glT, SQT);
    fprintf(f, " =================================================\n");
    fprintf(f, " R^2= %12e                        \n", R2);

    for (c = 0; c < r + 1; ++c)
        for (l = 0; l < s + 1; ++l) {
            fprintf(f, "+x^%d*y^%d*\t%12g\n", c, l, A[c + l * (r + 1)]);
        }

    fclose(f);
}
