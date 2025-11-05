#include <iostream>
#include <fstream>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>

using namespace std;

// 函式：生成矩陣
void gen_matrix(float**& matrix, int rows, int cols, int rank) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
            if (i == 0 && j == 0) {
                printf("Process %d on host '%s' (PID %d) generating matrix element (%d, %d) = %f with %d OpenMP threads\n",
                       rank, getenv("HOSTNAME"), getpid(), i, j, matrix[i][j], omp_get_max_threads());
            }
        }
    }
}

// 函式：輸出矩陣到檔案
void output_matrix(float** matrix, int rows, int cols, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << matrix[i][j] << " ";
        }
        file << endl;
    }
    file.close();
}

// 函式：矩陣乘法（使用 OpenMP）
void multiply_matrix(float** matrix_A, float** matrix_B, float**& matrix_C, int M, int N) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            matrix_C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                matrix_C[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
            if (i == 0 && j == 0) {
                printf("Process %d on host '%s' (PID %d) computed matrix_C element (%d, %d) = %f with %d OpenMP threads\n",
                       rank, getenv("HOSTNAME"), getpid(), i, j, matrix_C[i][j], omp_get_max_threads());
            }
        }
    }
}

int main(int argc, char** argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = 100, N = 100;
    if (argc > 1) {
        sscanf(argv[1], "%dx%d", &M, &N);
    }

    // 生成矩陣 A (M x N) 和 B (N x M)
    float** matrix_A = new float*[M];
    float** matrix_B = new float*[N];
    for (int i = 0; i < M; i++) {
        matrix_A[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        matrix_B[i] = new float[M];
    }

    gen_matrix(matrix_A, M, N, rank);
    gen_matrix(matrix_B, N, M, rank);

    // 輸出矩陣 A 和 B
    char filename_A[256], filename_B[256];
    sprintf(filename_A, "%d.matrix_%d.%d", rank, M, N);
    sprintf(filename_B, "%d.matrix_%d.%d", rank, N, M);
    output_matrix(matrix_A, M, N, filename_A);
    output_matrix(matrix_B, N, M, filename_B);

    // 矩陣乘法
    float** matrix_C = new float*[M];
    for (int i = 0; i < M; i++) {
        matrix_C[i] = new float[M];
    }
    multiply_matrix(matrix_A, matrix_B, matrix_C, M, N);

    // 輸出矩陣 C
    char filename_C[256];
    sprintf(filename_C, "%d.m-p_%d.%d", rank, M, M);
    output_matrix(matrix_C, M, M, filename_C);

    // 釋放記憶體
    for (int i = 0; i < M; i++) {
        delete[] matrix_A[i];
        delete[] matrix_C[i];
    }
    delete[] matrix_A;
    delete[] matrix_C;
    for (int i = 0; i < N; i++) {
        delete[] matrix_B[i];
    }
    delete[] matrix_B;

    MPI_Finalize();
    return 0;
}
