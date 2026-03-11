/*********************************************************************
 * 06_ompi_mutlip_matrix.c
 *
 * Hybrid OpenMPI + OpenMP demo: block‑row distributed matrix multiplication.
 *
 *  Usage:
 *      export OMP_NUM_THREADS=2 ; mpirun -np 2 ./06_ompi_mutlip_matrix.o matrix_A matrix_B
 *      export OMP_NUM_THREADS=4 ; mpirun -np 2 ./06_ompi_mutlip_matrix.o matrix_A matrix_B
 *
 *  Input files must be plain text, one row per line, columns separated
 *  by spaces (the format produced by ompi_gen_matrix.cpp).
 *
 *  The program prints, for every MPI rank and every OpenMP thread,
 *  the following information:
 *      - MPI rank / world size
 *      - host name
 *      - PID
 *      - OpenMP thread id / #threads
 *      - CPU core on which the thread runs
 *      - Row range owned by the thread (only for illustration)
 *
 *  After the multiplication the result matrix C (size M×K) is written
 *  by rank 0 to the file  <PID>.matrix_C_<M>.<K>
 *
 *  Compile (GCC‑based OpenMPI):
 *      mpicxx -std=c++11 -fopenmp -O2 06_ompi_mutlip_matrix.c -o 06_ompi_mutlip_matrix
 * 
 * ### LLM-AI ###
 * Module: GPT-OSS-120B
 * User prompt: 
 * 請撰寫一 c++ 程式(name : 06_ompi_mutlip_matrix.c)，用來展示 openmp 與 openmpi  多核心平行化、多執行緒、多分散式記憶體、平行計算之能力， 其目的計算矩陣乘法，遵守下列指引：
 * * 輸入為命令列指定兩個矩陣檔案 (file_A=matrix_A , file_B=matrix_B)
 * * 必須使用 openmp 與 openmpi ，並於矩陣產生過程中充分顯示使用 openmp (多核心 的平行化與多執行緒資訊)與 openmpi 狀況（如運行主機、核心編號、 PID）
 * 
 * * main 包括副函式：
 * ** this_matrix = multip_matrix(matrix_A,matrix_B) ：進行矩陣乘法，必須使用分塊分配，讓不同 rank 真正分工計算，再用 MPI 合併結果。必須充分使用 openmp 與 openmpi ，且用 #pragma omp critical 讓每個執行緒印，讓計算過程中充分顯示使用 openmp (多核平行化與多執行緒資訊)與 openmpi 分塊狀況（如運行主機、核心編號、 PID）
 * EOF:LLM-AI 
 *********************************************************************/

#include <mpi.h>
#include <omp.h>
#include <unistd.h>               // getpid()
#include <sched.h>                // sched_getcpu()
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cstring>                // strerror
#include <cstdlib>                // exit
#include <ctime>                  // time

/*------------------------------------------------------------------*/
/*  Return the CPU core the calling thread is currently executing on  */
/*------------------------------------------------------------------*/
static inline int current_cpu()
{
    return sched_getcpu();   // -1 on error (ignore for demo)
}

/*------------------------------------------------------------------*/
/*  Read a whitespace‑separated matrix file.                         */
/*  The function fills `mat` (row‑major), and returns rows/cols.      */
/*------------------------------------------------------------------*/
bool read_matrix(const std::string &filename,
                 std::vector<float> &mat,
                 int &rows, int &cols)
{
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Cannot open matrix file '" << filename << "'\n";
        return false;
    }

    std::string line;
    rows = 0;
    cols = -1;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;                 // skip empty lines
        std::istringstream iss(line);
        float v;
        std::vector<float> row_vals;
        while (iss >> v) row_vals.push_back(v);
        if (cols == -1) cols = static_cast<int>(row_vals.size());
        else if (static_cast<int>(row_vals.size()) != cols) {
            std::cerr << "Inconsistent column count in file '" << filename << "'\n";
            return false;
        }
        mat.insert(mat.end(), row_vals.begin(), row_vals.end());
        ++rows;
    }

    if (rows == 0 || cols == 0) {
        std::cerr << "Empty matrix in file '" << filename << "'\n";
        return false;
    }
    return true;
}

/*------------------------------------------------------------------*/
/*  Write a matrix to file  <PID>.matrix_C_<M>.<K>                    */
/*------------------------------------------------------------------*/
void write_matrix(const std::vector<float> &mat,
                 int rows, int cols,
                 int mpi_rank)                 // only rank 0 will call
{
    pid_t pid = getpid();
    std::ostringstream fname;
    fname << pid << ".matrix_C_" << rows << "." << cols;

    std::ofstream ofs(fname.str());
    if (!ofs) {
        std::cerr << "[MPI " << mpi_rank << "] cannot open output file "
                  << fname.str() << " (" << std::strerror(errno) << ")\n";
        return;
    }

    ofs << std::fixed << std::setprecision(6);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            ofs << mat[r * cols + c];
            if (c + 1 < cols) ofs << ' ';
        }
        ofs << '\n';
    }
    std::cout << "[MPI " << mpi_rank << "] PID " << pid
              << " wrote result matrix to " << fname.str() << std::endl;
}

/*--------------------------------------------------------------*/
/*  Multiply local block of A (rows_A_local × N) with full B (N × K)
    rows_A_local : 本 rank 本地持有的列數
    row_start   : 這些列在全域矩陣 A 中的起始列索引 (0‑based)                */
/*--------------------------------------------------------------*/
std::vector<float> multiply_block(const std::vector<float> &A_local,
                                  const std::vector<float> &B,
                                  int rows_A_local, int N, int K,
                                  int row_start,
                                  int mpi_rank, const char *host_name)
{
    pid_t pid = getpid();
    int nthreads = omp_get_max_threads();

    // ---------- 1. 印出「本 rank 處理哪幾列」 ----------
    int global_start = row_start;                       // 例如 0、16、32、48 …
    int global_end   = row_start + rows_A_local - 1;    // inclusive
    std::cout << "[MPI " << mpi_rank << "] PID " << pid
              << " starts multiplication (rows [" << global_start
              << " … " << global_end << "], N = " << N
              << ", K = " << K << "). OpenMP threads = "
              << nthreads << std::endl;

    std::vector<float> C_local(rows_A_local * K, 0.0f);

#pragma omp parallel
    {
        int tid   = omp_get_thread_num();
        int nthr  = omp_get_num_threads();
        int cpu   = current_cpu();

        /* -------------- 2. 印出每個 thread 的全域行範圍 -------------- */
#pragma omp critical
    {
        // 依 static 排程，將 rows_A_local 均分給所有執行緒。
        // 若 rows_A_local 無法被 nthr 整除，最後一個 thread 會多負責剩餘的行。
        int rows_per_thr = rows_A_local / nthr;
        int extra        = rows_A_local % nthr;          // 只能給前 few threads
        int local_start  = tid * rows_per_thr + std::min(tid, extra);
        int local_len    = rows_per_thr + (tid < extra ? 1 : 0);
        int local_end    = local_start + local_len - 1;   // inclusive

        // 轉換成全域索引
        int glob_start = row_start + local_start;
        int glob_end   = row_start + local_end;

        std::cout << "   [MPI " << mpi_rank << "] thread "
                  << tid << "/" << nthr
                  << " on CPU core " << cpu
                  << " → rows " << glob_start << "‑" << glob_end << std::endl;
    }

        /* 真正的乘法，使用 static 排程能保持前面的「行」切分 */
#pragma omp for schedule(static)
        for (int i = 0; i < rows_A_local; ++i) {          // each row of A_local
            for (int j = 0; j < K; ++j) {                // each column of B
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += A_local[i * N + k] * B[k * K + j];
                }
                C_local[i * K + j] = sum;
            }
        }
    }   // end omp parallel

    std::cout << "[MPI " << mpi_rank << "] PID " << pid
              << " finished its block multiplication.\n";

    return C_local;
}

/*------------------------------------------------------------------*/
/*  Main                                                            */
/*------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    /*-------------------- 1. Initialise MPI ------------------------*/
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char host_name[MPI_MAX_PROCESSOR_NAME];
    int host_len = 0;
    MPI_Get_processor_name(host_name, &host_len);
    host_name[host_len] = '\0';

    /*-------------------- 2. Check command line -------------------*/
    if (argc != 3) {
        if (world_rank == 0)
            std::cerr << "Usage: " << argv[0] << "  matrix_A  matrix_B\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::string file_A = argv[1];
    std::string file_B = argv[2];

    /*-------------------- 3. Rank 0 reads both matrices -------------*/
    int M = 0, N = 0, K = 0;                // A is M×N, B is N×K
    std::vector<float> A, B;

    if (world_rank == 0) {
        if (!read_matrix(file_A, A, M, N)) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (!read_matrix(file_B, B, N, K)) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (N != static_cast<int>(B.size() / K)) {
            std::cerr << "Dimension mismatch: A columns (" << N
                      << ") != B rows (" << N << ")\n";
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        std::cout << "=== MPI matrix multiplication demo ==================\n"
                  << "   Global sizes : A(" << M << "×" << N << "), "
                  << "B(" << N << "×" << K << ")\n"
                  << "   MPI size    : " << world_size << "\n"
                  << "   Host name   : " << host_name << "\n"
                  << "====================================================\n";
    }

    /*-------------------- 4. Broadcast dimensions -----------------*/
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /*-------------------- 5. Distribute A (block‑row) -------------*/
    // how many rows does this rank own?
    int rows_per_rank = M / world_size;
    int remainder     = M % world_size;                 // first `remainder` ranks get +1 row

    int local_rows = rows_per_rank + (world_rank < remainder ? 1 : 0);
    int row_start  = world_rank * rows_per_rank + std::min(world_rank, remainder);
    // number of floats this rank will receive
    int local_count = local_rows * N;
    std::vector<float> A_local(local_count);

    // Build sendcounts / displacements for Scatterv (only needed on root)
    std::vector<int> sendcounts, displs;
    if (world_rank == 0) {
        sendcounts.resize(world_size);
        displs.resize(world_size);
        for (int r = 0; r < world_size; ++r) {
            int r_rows = rows_per_rank + (r < remainder ? 1 : 0);
            sendcounts[r] = r_rows * N;
            displs[r] = (r == 0) ? 0
                                 : displs[r-1] + sendcounts[r-1];
        }
    }

    MPI_Scatterv( world_rank == 0 ? A.data() : nullptr,
                  world_rank == 0 ? sendcounts.data() : nullptr,
                  world_rank == 0 ? displs.data() : nullptr,
                  MPI_FLOAT,
                  A_local.data(),
                  local_count,
                  MPI_FLOAT,
                  0,
                  MPI_COMM_WORLD);

    /*-------------------- 6. Broadcast whole B to everybody --------*/
    if (world_rank != 0) B.resize(N * K);   // allocate space on non‑root ranks
    MPI_Bcast(B.data(), N * K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /*-------------------- 7. Multiply --------------------------------*/
    double t0 = MPI_Wtime();
    std::vector<float> C_local = multiply_block(A_local, B,
                                                local_rows, N, K,
                                                row_start,
                                                world_rank, host_name);
    double t1 = MPI_Wtime();

    if (world_rank == 0) {
        std::cout << "[MPI 0] Multiplication (including OpenMP) time = "
                  << (t1 - t0) << " s\n";
    }

    /*-------------------- 8. Gather the result matrix on rank 0 ---*/
    // Prepare receive buffers on root
    std::vector<int> recvcounts, recvdispls;
    if (world_rank == 0) {
        recvcounts.resize(world_size);
        recvdispls.resize(world_size);
        for (int r = 0; r < world_size; ++r) {
            int r_rows = rows_per_rank + (r < remainder ? 1 : 0);
            recvcounts[r] = r_rows * K;
            recvdispls[r] = (r == 0) ? 0
                                    : recvdispls[r-1] + recvcounts[r-1];
        }
    }

    std::vector<float> C_full;
    if (world_rank == 0) C_full.resize(M * K);

    MPI_Gatherv(C_local.data(),                // sendbuf
                local_rows * K, MPI_FLOAT,    // sendcount, type
                world_rank == 0 ? C_full.data() : nullptr,
                world_rank == 0 ? recvcounts.data() : nullptr,
                world_rank == 0 ? recvdispls.data() : nullptr,
                MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /*-------------------- 9. Output (only rank 0) -----------------*/
    if (world_rank == 0) {
        write_matrix(C_full, M, K, world_rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return EXIT_SUCCESS;
}