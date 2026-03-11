/*********************************************************************
 *  ompi_gen_matrix.cpp
 *
 *  Hybrid OpenMPI + OpenMP demo.
 *  - Generates ONE M×N matrix (float) in a block‑row distributed way.
 *  - Each MPI rank creates its rows in parallel with OpenMP.
 *  - Detailed thread‑creation information (rank, PID, thread‑id,
 *    #threads, CPU core, owned rows) is printed.
 *  - After generation the blocks are gathered on rank 0 (MPI_Gatherv)
 *    and written to a file named <PID>.matrix_<M>.<N>.
 *
 *  Compile (GCC‑based OpenMPI):
 *      mpicxx -std=c++11 -fopenmp -O2 ompi_gen_matrix.cpp -o ompi_gen_matrix
 *
 *  Run examples:
 *      export OMP_NUM_THREADS=2 ; mpirun -np 2 ./ompi_gen_matrix 32x64
 *
 *      export OMP_NUM_THREADS=8 ; mpirun -np 2 ./ompi_gen_matrix 32x64
 *      
 * ***************
 * 
 * ### LLM-AI generation ### 
 * Module : GPT-OSS-120B
 * User Prompt: 
請撰寫一 c++ 程式(name : ompi_gen_matrix.cpp)，用來展示   openmp 與 openmpi  多核心平行化、多執行緒、多分散式記憶體、平行計算之能力， 其目的為生成一組 M x N  實數(float 浮點數型態)矩陣，遵守下列指引：
* 輸入可於指令列參數指定一整數數對(M, N 如: 90x100 )；若無輸入，預設產生 100x100 之浮點數型態矩陣
* 必須使用 openmp 與 openmpi ，並於矩陣產生過程中充分顯示使用 openmp (多核心 的平行化與多執行緒資訊)與 openmpi 狀況（如運行主機、核心編號、 PID）
* main 包括副函式：
** this_matrix = gen_matrix(M,N) ：生成一 M x N 浮點數型態矩陣
** outout_matrix (matrix_A) : 輸出矩陣至檔案 $PID.matrix_$M.$N
ompi_gen_matrix 只會產生一組矩陣，gen_matrix 必須使用分塊分配，讓不同 rank 真正分工生成，再用 MPI_Reduce 合併結果。必須充分使用 openmp 與 openmpi ，並生成過程中充分顯示使用 openmp (多核平行化與多執行緒資訊)與 openmpi 狀況（如運行主機、核心編號、 PID）
## EOF:LLM-AI
 *********************************************************************/

#include <mpi.h>
#include <omp.h>
#include <unistd.h>               // getpid()
#include <sched.h>                // sched_getcpu()
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <sstream>
#include <cstring>                // strerror
#include <cstdlib>                // exit
#include <ctime>                  // time

/*------------------------------------------------------------------*/
/*  Return the CPU core the current thread is running on (Linux)    */
/*------------------------------------------------------------------*/
static inline int current_cpu()
{
    return sched_getcpu();   // -1 on error
}

/*------------------------------------------------------------------*/
/*  Parse MxN, M,X,N or M,N                                        */
/*------------------------------------------------------------------*/
bool parse_dimensions(const char *arg, int &M, int &N)
{
    std::string s(arg);
    for (char &c : s)
        if (c == 'x' || c == 'X' || c == ',')
            c = ' ';                     // turn delimiter into a space

    std::istringstream iss(s);
    // safe conversion to bool
    bool ok = static_cast<bool>(iss >> M >> N);
    return ok;
}

/*------------------------------------------------------------------*/
/*  Generate the *local* rows belonging to this MPI rank.           */
/*  Filling is parallelised with OpenMP.                           */
/*------------------------------------------------------------------*/
void gen_local_matrix(int M, int N,
                      int row_start, int local_rows,
                      std::vector<float> &local_mat,
                      int mpi_rank, int mpi_size,
                      const char *host_name)
{
    pid_t pid = getpid();

    /* ---- rank‑level description ----------------------------------- */
    std::cout << "[MPI " << mpi_rank << "/" << mpi_size
              << " on " << host_name << "] PID " << pid
              << " will generate rows [" << row_start
              << " … " << row_start + local_rows - 1 << "] "
              << "(" << local_rows << " rows) of a " << M << "×" << N
              << " matrix.   OMP threads = " << omp_get_max_threads()
              << std::endl;

    /* ---- parallel region ------------------------------------------ */
#pragma omp parallel
    {
        int tid   = omp_get_thread_num();          // 0 … #threads‑1
        int nthr  = omp_get_num_threads();
        int cpu   = current_cpu();

        /* ---- each thread prints its own info (critical → no interleaving) */
#pragma omp critical
        {
            /* compute which global rows this thread will **mostly** touch.
               We only need it for illustration; the real work is done by
               the omp for below.                                      */
            long long elems_per_thr = static_cast<long long>(local_rows) * N / nthr;
            long long start_elem = static_cast<long long>(tid) * elems_per_thr;
            long long end_elem   = (tid == nthr - 1)
                                   ? static_cast<long long>(local_rows) * N - 1
                                   : start_elem + elems_per_thr - 1;

            int start_row = row_start + static_cast<int>(start_elem / N);
            int   end_row = row_start + static_cast<int>(   end_elem / N);

            std::cout << "   [MPI " << mpi_rank << "] thread "
                      << tid << "/" << nthr
                      << " on CPU core " << cpu
                      << " → rows " << start_row << "‑" << end_row
                      << std::endl;
        }

        /* ---- random generator – one per thread (thread_local) -------- */
        static thread_local std::mt19937 rng(
            static_cast<unsigned>(std::time(nullptr)) + static_cast<unsigned>(pid) + static_cast<unsigned>(tid));
        static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        /* ---- actual filling ------------------------------------------ */
#pragma omp for schedule(static)
        for (int i = 0; i < local_rows * N; ++i)
            local_mat[i] = dist(rng);
    }   // end of omp parallel
}

/*------------------------------------------------------------------*/
/*  Write the *full* matrix (only rank 0 does this)                 */
/*------------------------------------------------------------------*/
void output_matrix(const std::vector<float> &mat,
                  int rows, int cols,
                  int mpi_rank)
{
    pid_t pid = getpid();
    std::ostringstream fname;
    fname << "gen05_" << pid << ".matrix_" << rows << "." << cols;

    std::ofstream ofs(fname.str());
    if (!ofs) {
        std::cerr << "[MPI " << mpi_rank << "] cannot open file "
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
              << " wrote matrix to file " << fname.str() << std::endl;
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

    /*-------------------- 2. Parse M and N -----------------------*/
    int M = 100, N = 100;                 // defaults
    if (argc >= 2) {
        if (!parse_dimensions(argv[1], M, N)) {
            if (world_rank == 0)
                std::cerr << "Usage: " << argv[0]
                          << " 90x100   (or 90,100)\n";
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (world_rank == 0) {
        std::cout << "=== Hybrid OpenMPI + OpenMP matrix generator ===\n"
                  << "   Global size : " << M << " × " << N << "\n"
                  << "   MPI size    : " << world_size << "\n"
                  << "   Host name   : " << host_name << "\n"
                  << "   -------------------------------------------\n";
    }

    /*-------------------- 3. Block‑row distribution -------------*/
    int rows_per_rank = M / world_size;
    int remainder     = M % world_size;               // extra rows

    int local_rows = rows_per_rank + (world_rank < remainder ? 1 : 0);
    int row_start  = world_rank * rows_per_rank
                     + std::min(world_rank, remainder); // first global row owned

    std::vector<float> local_matrix(local_rows * N);   // storage for this rank

    /*-------------------- 4. Generate local part -----------------*/
    double gen_start = MPI_Wtime();

    gen_local_matrix(M, N, row_start, local_rows,
                     local_matrix,
                     world_rank, world_size, host_name);

    double gen_end = MPI_Wtime();
    if (world_rank == 0)
        std::cout << "[MPI 0] Generation time (all ranks) = "
                  << (gen_end - gen_start) << " s\n";

    /*-------------------- 5. Gather on rank 0 -------------------*/
    int local_count = static_cast<int>(local_matrix.size());
    std::vector<int> recv_counts;   // only needed on root
    std::vector<int> displs;        // offsets (in floats)

    if (world_rank == 0) {
        recv_counts.resize(world_size);
        displs.resize(world_size);
    }

    MPI_Gather(&local_count, 1, MPI_INT,
               world_rank == 0 ? recv_counts.data() : nullptr,
               1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i)
            displs[i] = displs[i-1] + recv_counts[i-1];
    }

    std::vector<float> full_matrix;
    if (world_rank == 0)
        full_matrix.resize(M * N);

    MPI_Gatherv(local_matrix.data(), local_count, MPI_FLOAT,
                world_rank == 0 ? full_matrix.data() : nullptr,
                world_rank == 0 ? recv_counts.data() : nullptr,
                world_rank == 0 ? displs.data() : nullptr,
                MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /*-------------------- 6. Output (only rank 0) ---------------*/
    if (world_rank == 0)
        output_matrix(full_matrix, M, N, world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
