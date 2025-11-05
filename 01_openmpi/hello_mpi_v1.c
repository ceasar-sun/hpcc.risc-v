#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char** argv) {
    // 初始化 MPI 環境
    int rc;
    rc = MPI_Init(&argc, &argv);
    /* To check the return values on MPI calls */
    if (rc != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Init failed\n");
    }

    // 取得目前 process 的 ID（rank）
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 取得總 process 數量
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 取得主機名稱
    char hostname[256];
    int pid;
    gethostname(hostname, sizeof(hostname));
    pid = getpid();
    
    // 印出主機名稱與 "Hello World"
    printf("Hello from processor %d of %d ( pid : %d ,host : '%s')\n", rank, size, pid, hostname);
    fflush(stdout);

    // 結束 MPI 環境
    MPI_Finalize();
    return 0;
}

