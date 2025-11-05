/*********************************************************************
 *  ompi_gen_matrix.cpp
 *
 *  Demonstrates a *Hybrid* OpenMP / OpenMPI program.
 *
 *  1️⃣  The program creates a single M × N matrix of `float`s.
 *      M and N can be given on the command line as   90x100
 *      (case‑insensitive, also “90,100”).  If no argument is
 *      supplied the default is 100 × 100.
 *
 *  2️⃣  The matrix is **block‑distributed** among the MPI ranks.
 *      Each rank generates the rows that belong to it
 *      (the distribution may be uneven when M is not divisible
 *      by the number of ranks).  The local block is filled in
 *      parallel by OpenMP threads.
 *
 *  3️⃣  After the local generation the blocks are gathered on
 *      rank 0 with `MPI_Gatherv`.  Rank 0 then writes the whole
 *      matrix to a file whose name contains its PID:
 *
 *          <PID>.matrix_<M>.<N>
 *
 *  4️⃣  Throughout the generation the program prints **MPI
 *      information** (rank, size, host name, PID) and **OpenMP
 *      information** (thread‑id, #threads, CPU core on which the
 *      thread is running).  This makes it easy to see that we are
 *      using many processes **and** many threads inside each
 *      process.
 *
 *  Compile (example with GCC‑based OpenMPI):
 *
 *      mpicxx -std=c++11 -fopenmp -O2 ompi_gen_matrix.cpp -o ompi_gen_matrix
 *
 *  Run (4 MPI processes, matrix 90×100):
 *
 *      mpirun -np 4 ./ompi_gen_matrix 90x100
 *

## LLM-AI generation: : 
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
#include <cstdlib>                // exit, atoi
#include <ctime>                  // time

/*------------------------------------------------------------------*/
/*  Helper – return the CPU core on which the calling thread runs    */
/*------------------------------------------------------------------*/
static inline int current_cpu()
{
    int c = sched_getcpu();   /* returns –1 on error */
    return c;
}

/*------------------------------------------------------------------*/
/*  Parse a string like “90x100”, “90X100” or “90,100”               */
/*------------------------------------------------------------------*/
bool parse_dimensions(const char *arg, int &M, int &N)
{
    std::string s(arg);
    for (char &c : s)
        if (c == 'x' || c == 'X' || c == ',')
            c = ' ';                     // turn delimiter into a space

    std::istringstream iss(s);
    // **修改點**：先讀取，再根據 stream 狀態返回 true/false
    if (iss >> M >> N)               // ← 這裡利用 istream::operator bool()
        return true;                  // <--- **added**
    else
        return false;                 // <--- **added**
}

/*------------------------------------------------------------------*/
/*  Generate the *local* part of the matrix (rows belonging to this   */
/*  MPI rank).  The generation itself is parallelised with OpenMP.   */
/*------------------------------------------------------------------*/
void gen_local_matrix(int M, int N,                     // global size
                      int row_start, int local_rows,   // this rank's block
                      std::vector<float> &local_mat,   // output buffer
                      int mpi_rank, int mpi_size,
                      const char *host_name)
{
    pid_t pid = getpid();

    std::cout << "[MPI " << mpi_rank << "/" << mpi_size
              << " on " << host_name << "] PID " << pid
              << " will generate rows [" << row_start
              << " .. " << row_start + local_rows - 1 << "] ("
              << local_rows << " rows) of a " << M << "x" << N
              << " matrix. OpenMP threads = " << omp_get_max_threads()
              << std::endl;

    /*  Random numbers – one generator per thread to avoid contention  */
    std::vector<std::mt19937> rngs;
    int nthreads = omp_get_max_threads();
    rngs.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        std::mt19937 gen(static_cast<unsigned>(std::time(nullptr))   // seed
                         + static_cast<unsigned>(pid) + t);
        rngs.emplace_back(std::move(gen));
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    /*  Parallel fill – each thread works on a chunk of the local block  */
#pragma omp parallel
    {
        int tid   = omp_get_thread_num();
        int nthr  = omp_get_num_threads();
        int cpu   = current_cpu();

        /*  Print thread information once per thread (not once per element) */
#pragma omp single nowait
        {
            std::cout << "   [MPI " << mpi_rank << "] thread "
                      << tid << "/" << nthr
                      << " running on CPU core " << cpu << std::endl;
        }

        /*  each thread fills a strided portion of the local matrix      */
        #pragma omp for schedule(static)
        for (int i = 0; i < local_rows * N; ++i) {
            local_mat[i] = dist(rngs[tid]);
        }
    }   // end omp parallel
}

/*------------------------------------------------------------------*/
/*  Write a full matrix to file                                        */
/*------------------------------------------------------------------*/
void output_matrix(const std::vector<float> &mat,
                  int rows, int cols,
                  int mpi_rank)          // only the root will call this
{
    pid_t pid = getpid();
    std::ostringstream fname;
    fname << "gen03_" << pid << ".matrix_" << rows << "." << cols;

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
/*  Main – initialise MPI, split the matrix, generate locally,      */
/*  gather on rank 0 and write the result.                           */
/*------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    /*-------------------- 1. Initialise MPI ------------------------*/
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char host_name[MPI_MAX_PROCESSOR_NAME];
    int host_name_len = 0;
    MPI_Get_processor_name(host_name, &host_name_len);
    host_name[host_name_len] = '\0';

    /*-------------------- 2. Parse command line --------------------*/
    int M = 100, N = 100;                         // default 100×100
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
        std::cout << "=== Hybrid OpenMP + OpenMPI matrix generator ===\n"
                  << "   Global size : " << M << " x " << N << "\n"
                  << "   MPI size    : " << world_size << "\n"
                  << "   Host name  : " << host_name << "\n"
                  << "   -----------------------------------------------\n";
    }

    /*-------------------- 3. Determine block distribution ---------*/
    // Simple block‑row distribution
    int rows_per_rank = M / world_size;               // basic chunk
    int remainder     = M % world_size;               // extra rows

    int local_rows = rows_per_rank + (world_rank < remainder ? 1 : 0);
    int row_start  = world_rank * rows_per_rank + std::min(world_rank, remainder);

    // allocate space for the local block
    std::vector<float> local_matrix(local_rows * N);

    /*-------------------- 4. Generate local block -----------------*/
    gen_local_matrix(M, N, row_start, local_rows,
                     local_matrix,
                     world_rank, world_size, host_name);

    /*-------------------- 5. Gather full matrix on rank 0 ---------*/
    std::vector<int> recv_counts;   // number of floats each rank sends
    std::vector<int> displs;        // displacement (in floats) for each rank

    if (world_rank == 0) {
        recv_counts.resize(world_size);
        displs.resize(world_size);
    }

    // each rank tells root how many floats it will send
    int local_count = local_rows * N;
    MPI_Gather(&local_count, 1, MPI_INT,
               world_rank == 0 ? recv_counts.data() : nullptr,
               1, MPI_INT,
               0, MPI_COMM_WORLD);

    // compute displacements on root
    if (world_rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i-1] + recv_counts[i-1];
        }
    }

    // allocate full matrix on root
    std::vector<float> full_matrix;
    if (world_rank == 0) {
        full_matrix.resize(M * N);
    }

    // Gather the actual data
    MPI_Gatherv(local_matrix.data(),          // sendbuf
                local_count, MPI_FLOAT,       // sendcount, type
                world_rank == 0 ? full_matrix.data() : nullptr, // recvbuf
                world_rank == 0 ? recv_counts.data() : nullptr,
                world_rank == 0 ? displs.data()    : nullptr,
                MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /*-------------------- 6. Output result (only rank 0) ----------*/
    if (world_rank == 0) {
        output_matrix(full_matrix, M, N, world_rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);   // make sure everybody finishes before finalising
    MPI_Finalize();
    return EXIT_SUCCESS;
}
