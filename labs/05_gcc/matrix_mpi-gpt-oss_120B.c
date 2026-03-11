/*********************************************************************
 *  demo_omp_mpi_matrix.cpp
 *
 *  用 OpenMP + OpenMPI 示範單機多核心、 多執行緒、 多程序的平行運算。
 *
 *  功能：
 *   - 產生 M×N 矩陣 A、 N×M 矩陣 B
 *   - 把 A、B 寫入檔案   $PID.matrix_M.N   $PID.matrix_N.M
 *   - 計算 C = A × B (M×M) 並寫入檔案 $PID.m-p_M.M
 *
 *  每個 MPI rank 會以自己的 PID 為檔名前綴，並在所有平行區段
 *  印出 rank、PID、OpenMP thread id、所在 CPU core 等資訊。
 *
 *  編譯方式（以 MPI C++ 編譯器為例）：
 *      mpicxx -std=c++11 -fopenmp -O2 matrix_mpi-gpt-oss_120B.c -o matrix_mpi-gpt-oss_120B.o
 *
 *  執行方式（單機四核、啟動 4 個 MPI 程序）：
 *      mpirun -np 4 ./demo_matrix 90x100
 *
 *********************************************************************/
#include <mpi.h>
#include <omp.h>
#include <unistd.h>
#include <sched.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <ctime>

/*---------------------------------------------------------------*/
/*  取得當前 CPU core（Linux）                                    */
/*---------------------------------------------------------------*/
static inline int get_cpu_id()
{
    return sched_getcpu();   // 失敗時回傳 -1
}

/*---------------------------------------------------------------*/
/*  產生隨機浮點數矩陣 (rows × cols)                           */
/*---------------------------------------------------------------*/
void gen_matrix(int rows, int cols,
                std::vector<float> &mat,
                int mpi_rank, int /*mpi_size*/,
                const char *matrix_name)
{
    const int num_threads = omp_get_max_threads();
    const pid_t pid = getpid();

    std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)) + mpi_rank);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::cout << "[MPI " << mpi_rank << "] PID " << pid
              << " start generating " << matrix_name
              << " (" << rows << "x" << cols << "), OMP threads = "
              << num_threads << std::endl;

    /* ---------------------------------------------------------
       把「只印一次」的訊息放在 parallel 區段的 single 中，
       再把實際的填值交給 parallel for（這是合法的結構）。
       --------------------------------------------------------- */
#pragma omp parallel
    {
#pragma omp single
        {
            int tid = omp_get_thread_num();   // 此時只有一個 thread（0）
            int cpu = get_cpu_id();
            std::cout << "  [MPI " << mpi_rank << "] PID " << pid
                      << " Thread " << tid << "/" << num_threads
                      << " on CPU core " << cpu << std::endl;
        }

#pragma omp for schedule(static)
        for (int i = 0; i < rows * cols; ++i)
            mat[i] = dist(rng);
    }
}

/*---------------------------------------------------------------*/
/*  把矩陣寫成檔案                                            */
/*---------------------------------------------------------------*/
void output_matrix(const std::vector<float> &mat,
                  int rows, int cols,
                  int mpi_rank)
{
    pid_t pid = getpid();
    std::ostringstream fname;
    fname << pid << ".matrix_" << rows << "." << cols;

    std::ofstream ofs(fname.str());
    if (!ofs)
    {
        std::cerr << "[MPI " << mpi_rank << "] cannot open file "
                  << fname.str() << " (" << std::strerror(errno) << ")\n";
        return;
    }

    ofs << std::fixed << std::setprecision(6);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            ofs << mat[r * cols + c];
            if (c + 1 < cols) ofs << ' ';
        }
        ofs << '\n';
    }

    std::cout << "[MPI " << mpi_rank << "] PID " << pid
              << " wrote file " << fname.str() << std::endl;
}

/*---------------------------------------------------------------*/
/*  矩陣乘法 C = A (M×N) * B (N×M) → C (M×M)                */
/*---------------------------------------------------------------*/
std::vector<float> multiply_matrix(const std::vector<float> &A,
                                   const std::vector<float> &B,
                                   int M, int N,
                                   int mpi_rank)
{
    const int num_threads = omp_get_max_threads();
    const pid_t pid = getpid();

    std::cout << "[MPI " << mpi_rank << "] PID " << pid
              << " start multiplication (A:" << M << "x" << N
              << " B:" << N << "x" << M << "), OMP threads = "
              << num_threads << std::endl;

    std::vector<float> C(M * M, 0.0f);

#pragma omp parallel
    {
#pragma omp single
        {
            int tid = omp_get_thread_num();
            int cpu = get_cpu_id();
            std::cout << "  [MPI " << mpi_rank << "] PID " << pid
                      << " Thread " << tid << "/" << num_threads
                      << " on CPU core " << cpu << std::endl;
        }

#pragma omp for schedule(static)
        for (int i = 0; i < M; ++i)          // 行
        {
            for (int j = 0; j < M; ++j)      // 列
            {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k)
                    sum += A[i * N + k] * B[k * M + j];
                C[i * M + j] = sum;
            }
        }
    }

    std::cout << "[MPI " << mpi_rank << "] PID " << pid
              << " multiplication finished.\n";
    return C;
}

/*---------------------------------------------------------------*/
/*  解析 MxN、MxN、M,N 等字串                                 */
/*---------------------------------------------------------------*/
bool parse_dimensions(const char *arg, int &M, int &N)
{
    // 支援 "90x100", "90X100", "90,100"
    std::string s(arg);
    for (char &c : s)
        if (c == 'x' || c == 'X' || c == ',')
            c = ' ';

    std::istringstream iss(s);
    // 先讀入 M、N，然後檢查 stream 狀態是否 good()
    if (iss >> M >> N)
        return true;
    else
        return false;
}

/*---------------------------------------------------------------*/
/*  主程式                                                       */
/*---------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char host_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    MPI_Get_processor_name(host_name, &name_len);
    host_name[name_len] = '\0';

    int M = 100, N = 100;                 // 預設 100x100
    if (argc >= 2)
    {
        if (!parse_dimensions(argv[1], M, N))
        {
            if (world_rank == 0)
                std::cerr << "Usage: " << argv[0] << " 90x100   (or 90,100)\n";
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (world_rank == 0)
    {
        std::cout << "=== OpenMP + OpenMPI matrix demo =====================\n"
                  << "  host: " << host_name << "\n"
                  << "  MPI size: " << world_size << "\n"
                  << "  matrix size: A(" << M << "x" << N << "), "
                  << "B(" << N << "x" << M << ")\n"
                  << "====================================================\n";
    }

    // ----------------------------------------------
    // 每個 rank 自己產生 A、B、寫檔、相乘、寫檔
    // ----------------------------------------------
    std::vector<float> A(M * N);
    std::vector<float> B(N * M);

    gen_matrix(M, N, A, world_rank, world_size, "A");
    gen_matrix(N, M, B, world_rank, world_size, "B");

    output_matrix(A, M, N, world_rank);
    output_matrix(B, N, M, world_rank);

    std::vector<float> C = multiply_matrix(A, B, M, N, world_rank);

    // 輸出 C
    {
        pid_t pid = getpid();
        std::ostringstream fname;
        fname << pid << ".m-p_" << M << "." << M;

        std::ofstream ofs(fname.str());
        if (!ofs)
        {
            std::cerr << "[MPI " << world_rank << "] cannot open file "
                      << fname.str() << " (" << std::strerror(errno) << ")\n";
        }
        else
        {
            ofs << std::fixed << std::setprecision(6);
            for (int r = 0; r < M; ++r)
            {
                for (int c = 0; c < M; ++c)
                {
                    ofs << C[r * M + c];
                    if (c + 1 < M) ofs << ' ';
                }
                ofs << '\n';
            }
            std::cout << "[MPI " << world_rank << "] PID " << pid
                      << " wrote result file " << fname.str() << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0)
        std::cout << "=== All MPI ranks finished ===\n";

    MPI_Finalize();
    return EXIT_SUCCESS;
}
