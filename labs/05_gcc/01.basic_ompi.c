#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>

int main(int argc, char** argv) {
    // 初始化 MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 取得主機名稱
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    // OpenMP 相關資訊
    int max_threads = omp_get_max_threads();
    int num_procs = omp_get_num_procs();

    printf("Process %d/%d running on host '%s'\n", rank, size, hostname);
    printf("  - Maximum OpenMP threads available: %d\n", max_threads);
    printf("  - Number of processors available: %d\n", num_procs);

    // 每個 MPI 進程內，輸出當前的執行緒數
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        printf("  - Hello from MPI process %d/%d, thread %d/%d\n", rank, size, thread_id, num_threads);
    }

    // 結束 MPI
    MPI_Finalize();
    return 0;
}
