/*********************************************************************
 *  ompi_gen_matrix.cpp
 *
 *  Hybrid OpenMPI + OpenMP demo.
 *  - Generates ONE M×N matrix (float) in a *block‑row* distributed way.
 *  - Each MPI rank creates its local rows in parallel with OpenMP.
 *  - Detailed thread creation information is printed:
 *        rank, PID, OpenMP thread id, #threads, CPU core, local row range
 *  - After generation the blocks are gathered on rank 0 (MPI_Gatherv)
 *    and written to a file named <PID>.matrix_<M>.<N>.
 *
 *  Compile (GCC‑based OpenMPI):
 *      mpicxx -std=c++11 -fopenmp -O2 ompi_gen_matrix.cpp -o ompi_gen_matrix
 *
 *  Run examples:
 *      export OMP_NUM_THREADS=2            # 2 threads per MPI process
 *      mpirun -np 4 ./ompi_gen_matrix 90x100
 *
 *      export OMP_NUM_THREADS=8            # 8 threads per MPI process
 *      mpirun -np 4 ./ompi_gen_matrix 90x100
 *
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
    int c = sched_getcpu();      // -1 on error
    return c;
}

/*------------------------------------------------------------------*/
/*  Parse the command‑line argument “90x100”, “90,100”, …           */
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
/*  Generate the *local* rows belonging to this MPI rank.           */
/*  The filling loop itself is parallelised with OpenMP.            */
/*------------------------------------------------------------------*/
void gen_local_matrix(int M, int N,
                      int row_start, int local_rows,
                      std::vector<float> &local_mat,
                      int mpi_rank, int mpi_size,
                      const char *host_name)
{
    pid_t pid = getpid();

    /* ------------------ 1. 先印出本 rank 的概況 ------------------ */
    std::cout << "[MPI " << mpi_rank << "/" << mpi_size
              << " on " << host_name << "] PID " << pid
              << " will generate rows [" << row_start
              << " … " << row_start + local_rows - 1 << "] "
              << "(" << local_rows << " rows) of a " << M << "×" << N
              << " matrix.   OMP threads = " << omp_get_max_threads()
              << std::endl;

    /* --------------------------------------------------------------
       2. 每個 OpenMP 執行緒只印一次自己的資訊（包含 CPU core）。
          使用 `single nowait` 讓所有執行緒都能同時走過，
          但每個執行緒只會進入這個區塊一次。
       -------------------------------------------------------------- */
#pragma omp parallel
    {
        int tid   = omp_get_thread_num();          // 0 … #threads‑1
        int nthr  = omp_get_num_threads();         // total threads in this team
        int cpu   = current_cpu();

#pragma omp single nowait
        {
            /* 計算這個執行緒在「行」層面的負責區間（僅供顯示，實際填值仍交給 for-loop）。
               使用簡單的循環切分方式：每個 thread 處理相同數量的元素（static schedule）。
               只要顯示一次即可，避免訊息爆炸。 */
            int elems_per_thread = (local_rows * N) / nthr;
            int start_elem = tid * elems_per_thread;
            int end_elem   = (tid == nthr - 1) ? (local_rows * N) - 1
                                               : start_elem + elems_per_thread - 1;

            int start_row = row_start + start_elem / N;   // 轉成全球 row 編號
            int end_row   = row_start + end_elem   / N;

            std::cout << "   [MPI " << mpi_rank << "] thread "
                      << tid << "/" << nthr
                      << " on CPU core " << cpu
                      << " → rows " << start_row << "‑" << end_row
                      << std::endl;
        }

        /* ---------------------------------------------------------
           3. 真正的填值工作。使用 `omp for`（static）讓每個執行緒自行
              把自己的子區間寫滿。這裡不再使用 `single`，避免
              「work‑sharing nested inside work‑sharing」的錯誤。
           --------------------------------------------------------- */
#pragma omp for schedule(static)
        for (int i = 0; i < local_rows * N; ++i) {
            // 每個執行緒使用自己的 RNG（seed依tid區分，避免競爭）
            static thread_local std::mt19937 rng(
                static_cast<unsigned>(std::time(nullptr)) + static_cast<unsigned>(pid) + static_cast<unsigned>(tid));
            static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            local_mat[i] = dist(rng);
        }
    }   // end omp parallel
}

/*------------------------------------------------------------------*/
/*  Write the *full* matrix (only rank 0 calls this)                */
/*------------------------------------------------------------------*/
void output_matrix(const std::vector<float> &mat,
                  int rows, int cols,
                  int mpi_rank)
{
    pid_t pid = getpid();
    std::ostringstream fname;
    fname << pid << ".matrix_" << rows << "." << cols;

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
    int M = 100, N = 100;                 // default
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
    // simple block‑row splitting (may be uneven when M % world_size != 0)
    int rows_per_rank = M / world_size;
    int remainder     = M % world_size;                // extra rows

    int local_rows = rows_per_rank + (world_rank < remainder ? 1 : 0);
    int row_start  = world_rank * rows_per_rank
                     + std::min(world_rank, remainder);   // first global row owned

    std::vector<float> local_matrix(local_rows * N);    // storage for our block

    /*-------------------- 4. Generate local part -----------------*/
    // optional timing of the generation step
    double gen_start = MPI_Wtime();

    gen_local_matrix(M, N, row_start, local_rows,
                     local_matrix,
                     world_rank, world_size, host_name);

    double gen_end = MPI_Wtime();
    if (world_rank == 0)
        std::cout << "[MPI 0] Generation time (all ranks) = "
                  << (gen_end - gen_start) << " s\n";

    /*-------------------- 5. Gather on rank 0 -------------------*/
    // First inform rank 0 how many floats each rank will send
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

    // Compute displacements on root
    if (world_rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i)
            displs[i] = displs[i-1] + recv_counts[i-1];
    }

    // Allocate full matrix on root
    std::vector<float> full_matrix;
    if (world_rank == 0)
        full_matrix.resize(M * N);

    // Gather the actual data
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
