
## 📋 索引

### 一、說明
- [OpenMP / OpenMPI：概念、差異與典型用途](#-OpenMP / OpenMPI：概念、差異與典型用途)
- [OpenMP 與 OpenMPI  簡要差異](#-OpenMP 與 OpenMPI  簡要差異)

---

## OpenMP / OpenMPI：概念、差異與典型用途

| 項目 | **OpenMP** | **OpenMPI** |
|------|------------|-------------|
| **全名** | Open Multi‑Processing | Open Message Passing Interface (MPI) 的一個開放源碼實作 |
| **屬於哪一層** | **共享記憶體平行**（Shared‑memory）<br>在單一個節點（Node）內，多核心／多執行緒間共享同一塊記憶體空間 | **分散式記憶體平行**（Distributed‑memory）<br>跨多個節點（Node）透過訊息傳遞（Message Passing）交換資料 |
| **平行模型** | **Fork‑Join**：程式在「平行區塊」被 fork 成多個 thread，執行完畢再 join 回原始 thread | **SPMD（Single Program Multiple Data）**：每個進程都從同一個程式開始執行，透過 MPI API 明確地發送/接收訊息 |
| **程式語言支援** | C / C++ / Fortran（標準化的 `#pragma omp`、`!$OMP` 指示詞） | C / C++ / Fortran（MPI 標準函式，如 `MPI_Init`, `MPI_Send`, `MPI_Recv`）以及多種語言的绑定（Python: `mpi4py`、Julia: `MPI.jl` 等） |
| **編譯/鏈結** | 只需要在編譯時加上 OpenMP 旗標（GCC: `-fopenmp`、Intel: `-qopenmp`、Clang: `-fopenmp`） | 需要使用 MPI 編譯器包（如 `mpicc`, `mpicxx`, `mpif90`），這些包會自動加入正確的 include、library 路徑 |
| **執行環境** | 同一台機器（或同一個 NUMA 節點）內的 CPU 核心、甚至是同一台機器上的 GPU （使用 `omp target`） | 任意數量的節點（單機多卡、集群、雲端叢集），每個節點可以是多核心 CPU、GPU，甚至是異構硬體 |
| **記憶體模型** | **一致性記憶體**：所有 thread 直接存取同一塊全域記憶體，需要使用同步原語（critical、atomic、lock、barrier）避免競爭 | **私有記憶體**：每個 MPI 進程有自己的地址空間，資料必須透過 `MPI_Send/Recv`, `MPI_Bcast`, `MPI_Reduce` 等函式顯式搬運 |
| **適合的演算法** | 共享資料結構、迴圈平行化、簡單的工作分配（如 `#pragma omp parallel for`） | 大規模分布式演算法、需要跨節點的資料分割（Domain Decomposition）、所有需要明確通訊的情況（如 CFD、分子動力學、圖演算法） |
| **典型使用者** | 科學家、工程師、資料分析師想要快速把原本的串列程式在多核心 CPU 上提升 2–8 倍 | HPC 系統管理員、超算使用者、需要在多節點叢集上跑大規模模擬（1000+ 核心）的人 |
| **部署與管理** | 無需排程系統，直接執行即可（除非在叢集上使用 `srun`/`mpirun` 且結合 MPI） | 常與排程系統（Slurm、PBS、LSF）一起使用，透過 `srun`, `mpirun`, `mpiexec` 指定節點與任務數量 |
| **優點** | - 實作簡單（只加 few pragmas）<br>- 無需重新編寫程式的主要結構<br>- 編譯器自動處理負載平衡、thread 建立 | - 能擴展到上千、上萬核心<br>- 完全控制資料流向與通訊模式<br>- 跨平台、跨硬體廠商（Intel, Cray, IBM, NERSC, AWS, Azure） |
| **缺點** | - 只能在共享記憶體機器上使用，規模受限於單節點的核心數<br>- 共享記憶體競爭（cache 效率、false sharing） | - 程式撰寫較複雜，需要手動分割資料與管理通訊<br>- Debug、效能分析成本較高 |
| **常見的混合使用** | **MPI + OpenMP**（Hybrid）: 每個 MPI 進程僅使用節點內部的多個 core（OpenMP）<br>可在大型叢集上減少 MPI 任務數、降低通訊瓶頸 | 同上，常見於大型超算（例如 Cray XC 系列、Intel Xeon + NVIDIA GPU 計算節點） |

---

## 1. OpenMP：共享記憶體平行的「指示詞」方式

### 1.1 基本概念
- **fork‑join 模式**：程式執行到 `#pragma omp parallel` 之後，執行緒（thread）被「fork」出來，形成一個平行區塊；執行完後所有執行緒「join」回原始執行緒。
- **工作分配**：最常見的是 `parallel for`（或 `omp parallel for`）自動將迴圈迭代分配給不同 thread。
- **同步原語**：`critical`, `atomic`, `barrier`, `lock`, `flush` 等保證共享資源不發生競爭。

### 1.2 範例（C 語言）

```c
#include <stdio.h>
#include <omp.h>

int main(void) {
    const int N = 1000;
    double a[N], b[N], c[N];

    /* 初始化向量 */
    for (int i = 0; i < N; ++i) {
        a[i] = i * 1.0;
        b[i] = (N - i) * 0.5;
    }

    /* 向量加法 – OpenMP 平行化 */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }

    printf("c[0] = %f, c[N-1] = %f\n", c[0], c[N-1]);
    return 0;
}
```

- **編譯**（GCC）：`gcc -fopenmp -O2 vec_add.c -o vec_add`
- **執行**：`export OMP_NUM_THREADS=8 ./vec_add`（設定欲使用的 thread 數）

### 1.3 常見指示詞（OpenMP 5.0+）

| 指示詞 | 功能 | 範例 |
|--------|------|------|
| `parallel` | 建立平行區塊 | `#pragma omp parallel` |
| `for` / `parallel for` | 並行化迴圈 | `#pragma omp parallel for` |
| `sections` / `section` | 把不同程式段落分配給不同 thread | `#pragma omp sections` |
| `single` | 只讓一個 thread 執行 (常用於 I/O) | `#pragma omp single` |
| `critical` | 確保一次只有一個 thread 進入 | `#pragma omp critical` |
| `atomic` | 對單一變數的原子操作 | `#pragma omp atomic` |
| `barrier` | 所有 thread 必須同步 | `#pragma omp barrier` |
| `task` / `taskwait` | 動態任務生成（適合不規則工作） | `#pragma omp task` |
| `declare reduction` | 自訂 reduction 操作 | `#pragma omp declare reduction` |
| `target` / `teams` / `distribute` | 針對 GPU/加速器的指示（OpenMP offload） | `#pragma omp target` |

### 1.4 優化要點

1. **CPU 綁定（affinity）**：`export OMP_PROC_BIND=close` 能避免 thread 在不同 core 之間跳動，提升 cache 命中率。
2. **排程策略**：`static`（預先均分，開銷低） vs `dynamic`（工作量不均時使用） vs `guided`（逐步縮小 chunk）。
3. **避免 false sharing**：確保每個 thread 存取的資料行（cache line）不重疊，可用 padding 或 `alignas(64)`。
4. **向量化（SIMD）**：OpenMP 4.0+ 支援 `#pragma omp simd`，讓編譯器在迴圈內利用 AVX‑512、NEON 等向量指令。

---

## 2. OpenMPI（MPI）：分散式記憶體的訊息傳遞

### 2.1 基本概念
- **進程（process）**：每個 MPI 執行體是一個獨立的作業系統進程，擁有自己的虛擬地址空間。
- **通訊介面**：MPI 標準提供點對點（`MPI_Send`, `MPI_Recv`）與集合通信（`MPI_Bcast`, `MPI_Reduce`, `MPI_Alltoall`）等函式。
- **通訊域（Communicator）**：預設的 `MPI_COMM_WORLD` 包含所有啟動的 MPI 進程；也可自訂子集合（`MPI_Comm_split`）。
- **排程與執行**：在叢集上，使用 `mpirun`/`mpiexec`（或排程系統的 `srun`）指定節點與每節點的進程數。

### 2.2 範例（C 語言）——「Hello World」

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);                 // 初始化 MPI

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // 取得本進程 rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // 取得總進程數

    printf("Hello world from rank %d of %d\n", world_rank, world_size);

    MPI_Finalize();                         // 結束 MPI
    return 0;
}
```

- **編譯**（使用 OpenMPI）：`mpicc -O2 hello_mpi.c -o hello_mpi`
- **執行**（2 個進程）：`mpirun -np 2 ./hello_mpi`

### 2.3 範例（C）——「分布式矩陣乘法（行分塊）」

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void random_fill(double *A, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        A[i] = drand48();
}

/* 只在每個 rank 上分配自己的子矩陣 A_block，B 為廣播的完整矩陣 */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1024;                    // 矩陣大小 (N x N)
    const int rows_per_rank = N / size;    // 假設 N 能被 size 整除

    double *A_block = malloc(rows_per_rank * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C_block = malloc(rows_per_rank * N * sizeof(double));

    /* 每個 rank 產生自己的 A_block 隨機數 */
    srand48(rank + 1);
    random_fill(A_block, rows_per_rank, N);

    /* rank 0 產生 B 並廣播給所有 rank */
    if (rank == 0) random_fill(B, N, N);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* 本地矩陣乘法 A_block * B -> C_block */
    for (int i = 0; i < rows_per_rank; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A_block[i * N + k] * B[k * N + j];
            C_block[i * N + j] = sum;
        }
    }

    /* 收集所有 C_block 到 rank 0（可選） */
    double *C = NULL;
    if (rank == 0) C = malloc(N * N * sizeof(double));
    MPI_Gather(C_block, rows_per_rank * N, MPI_DOUBLE,
               C, rows_per_rank * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("C[0,0] = %f\n", C[0]);
        free(C);
    }

    free(A_block);
    free(B);
    free(C_block);
    MPI_Finalize();
    return 0;
}
```

- **編譯**：`mpicc -O3 -march=native matmul_mpi.c -o matmul_mpi`
- **執行（4 節點）**：`mpirun -np 4 ./matmul_mpi`

這個範例展示了 **資料分塊**（Block Row Decomposition）與 **集合通訊**（`MPI_Bcast`, `MPI_Gather`）的典型流程。

### 2.4 常見 MPI 功能與對應場景

| 功能 | 典型使用情境 |
|------|-------------|
| `MPI_Send` / `MPI_Recv`（阻塞）| 小量點對點訊息、同步資料傳遞 |
| `MPI_Isend` / `MPI_Irecv`（非阻塞）| 重疊通訊與運算、提升可伸縮性 |
| `MPI_Bcast` | 把同一筆資料（例如係數矩陣）廣播給所有進程 |
| `MPI_Reduce` / `MPI_Allreduce` | 計算全局總和、最大值、最小值等歸約操作 |
| `MPI_Scatter` / `MPI_Gather` | 把大資料分散到各進程、再收集結果 |
| `MPI_Alltoall` | 所有進程彼此交換同等大小的子集合（如 FFT 的步驟） |
| `MPI_Comm_split` | 建立子 Communicator，做分組或階層式通訊 |
| `MPI_Win`, `MPI_Put/Get`（RMA）| 一段式遠端記憶體存取，適合 PGAS（Partitioned Global Address Space）模型 |

### 2.5 選擇 MPI 的情況

- **規模大於單節點核心數**：需要跨機器、跨 rack、甚至跨資料中心的計算。
- **資料結構需要明確劃分**：例如 CFD 使用 3‑D 網格分塊、天體模擬使用空間分割。
- **需要確保計算結果的可重現性**：MPI 的決定性通訊讓結果較容易驗證（相較於某些非同步 task 系統）。
- **超級電腦或雲端叢集**：大多數 HPC 環境已經預裝 MPI，且排程系統與 MPI 整合緊密。

---

## 3. Hybrid 使用：MPI + OpenMP（或 MPI + CUDA）

### 為什麼要混合？

| 問題 | 單一模型的局限 | 混合模型的解法 |
|------|---------------|----------------|
| **節點內核數過多**（如 64‑core CPU + 8 GPU） | MPI 每個核心分配一個進程會產生過多的 MPI 任務、通訊開銷上升 | 在每個節點內部利用 OpenMP（或 CUDA）多執行緒／GPU，使 MPI 任務數量減少 |
| **記憶體瓶頸** | 每個 MPI 進程都有自己的複製記憶體，可能導致 RAM 使用過高 | OpenMP 共享同一片記憶體，降低複製成本 |
| **網路拓撲不均**（如多層交換機） | 全部 MPI 產生大量小訊息，網路負載不均 | 把局部密集運算留給 OpenMP，在節點間只傳遞必要的邊界資料 |

### 基本範例（Hybrid C 程式）

```c
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int N = 4096;
    const int local_N = N / world_size;   // 每個 MPI 只有 N/world_size 行

    double *A = malloc(local_N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(local_N * N * sizeof(double));

    /* 初始化 A 與 B（簡化為全部 rank 都產生相同的 B） */
    for (int i = 0; i < local_N * N; ++i) A[i] = drand48();
    if (world_rank == 0) for (int i = 0; i < N * N; ++i) B[i] = drand48();
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* OpenMP 內部平行化 */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }

    /* 可以再用 MPI 收集結果或直接寫出檔案 */
    MPI_Finalize();
    free(A); free(B); free(C);
    return 0;
}
```

- **編譯**（OpenMPI + OpenMP）：`mpicc -fopenmp -O3 hybrid_matmul.c -o hybrid_matmul`
- **執行**（每節點 2 個 MPI 任務、每任務 8 個 OpenMP thread）：  
  `mpirun -np 4 -bind-to core -map-by node:PE=8 ./hybrid_matmul`

---

## 4. 如何在實務上決定使用哪一個？

| **決策因素** | **若是** → 用 **OpenMP** | **若是** → 用 **MPI** |
|--------------|------------------------|----------------------|
| **硬體規模** | 單機多核、NUMA 內部 | 多節點、跨機箱、跨叢集 |
| **演算法特性** | 迴圈平行、共享陣列、資料依賴簡單 | domain decomposition、需要明確資料交換 |
| **開發時間** | 想快速加速已有串列程式 | 已接受訊息傳遞概念、想要伸縮到上千核心 |
| **記憶體需求** | 需要大量共享資料（避免複製） | 每個節點記憶體足夠，且資料可切分 |
| **效能瓶頸** | 記憶體頻寬、cache 效率 | 網路帶寬、通訊延遲 |
| **維護成本** | 只加 pragmas、相容性好 | 設計通訊協定、測試多種拓撲 |
| **未來擴展** | 可能升級到 2–4 個 CPU（仍在同一機箱） | 計畫將程式搬到 HPC 叢集或雲端 |

> **小技巧**：先用 OpenMP 把程式加速到 8‑16 核心，觀察效能提升。若仍未達到目標，再考慮把資料切分、加入 MPI，形成 **Hybrid** 架構。

---

## 5. 常見工具與資源

| 類別 | 工具 / 套件 | 用途 |
|------|------------|------|
| **編譯器** | GCC (`-fopenmp`), Intel ICC (`-qopenmp`), Clang (`-fopenmp`), NVCC (CUDA＋OpenMP), `mpicc`/`mpicxx` | 產生支援 OpenMP / MPI 的執行檔 |
| **除錯** | `gdb` + `-g -O0`, `TotalView`, `Allinea DDT` | 逐步除錯 MPI 程式（支援多進程） |
| **效能分析** | **VTune Amplifier**, **Intel Advisor**, **Perf**, **MPI‑P**, **TAU**, **Vampir**, **HPCToolkit**, **Nsight Systems** | 測量 CPU/GPU 使用率、通訊時間、記憶體存取 |
| **排程系統** | Slurm, PBS/Torque, LSF, SGE | 申請節點資源、提交 MPI / Hybrid 作業 |
| **容器** | Docker, Singularity / Apptainer | 把 MPI / OpenMP 環境封裝，保證可搬移性 |
| **雲端 HPC** | AWS ParallelCluster, Azure CycleCloud, Google Cloud HPC | 按需求租用多節點叢集，測試 MPI 程式 |
| **教學資源** | 《Using MPI—A Tutorial》, 《OpenMP Application Programming Interface》, 《MPI: The Complete Reference》, Coursera “Parallel Programming in C with MPI and OpenMP”, NTU HPC課程, Scalable Computing Lab (SCL) slides | 系統化學習與範例程式 |

---

## 6. 小結

| 項目 | OpenMP | OpenMPI |
|------|--------|----------|
| **平行模型** | 共享記憶體、fork‑join、執行緒 | 分散記憶體、進程、訊息傳遞 |
| **適用範圍** | 單機、多核／NUMA、GPU offload（OpenMP 5） | 多節點叢集、超算、雲端 HPC |
| **開發門檻** | 低（加 pragmas） | 中等（設計通訊、領域分割） |
| **可擴展性** | 受限於單節點記憶體與核心數 | 幾乎無上限（視硬體與網路） |
| **常見組合** | OpenMP + SIMD + GPU offload | MPI + OpenMP + CUDA (Hybrid) |
| **典型應用** | 向量/矩陣運算、圖像處理、機器學習前處理 | CFD、天文模擬、分子動力學、深度學習分布式訓練 |


---

# OpenMP 與 OpenMPI  簡要差異

---  

## **OpenMP 與 OpenMPI**  
在高效能運算 (HPC) 領域，OpenMP 與 OpenMPI 是兩種常見的平行計算工具，但它們的目的、設計理念、以及使用場景完全不同。

---  

### **1. OpenMP**  

#### **什麼是 OpenMP？**  

* OpenMP (Open Multi‑Processing) 是一個用於 **共享記憶體** 架構的多執行緒平行程式設計介面。  
* 它主要用於 **單機多核心** 的平行化，利用多個 CPU 核心同時執行程式碼的不同部分。  
* OpenMP 透過編譯器指令 (pragma) 來控制平行化，例如 `#pragma omp parallel for`。  

#### **特點**  

| 特點 | 說明 |
|------|------|
| 共享記憶體 | 所有執行緒共享同一個記憶體空間。 |
| 多執行緒 | 程式內部透過執行緒來平行執行任務。 |
| 易用性 | 只需在程式碼中加入 OpenMP 指令，編譯器會自動處理多執行緒。 |
| 效能 | 因為是共享記憶體，資料交換與同步的開銷較低。 |

#### **適用場景**  

* 單機多核心運算：矩陣運算、線性代數、科學模擬等。  
* 迴圈平行化：將重複的迴圈分配到多個核心上執行。  
* 多執行緒任務：適合 CPU 密集型的運算。  

#### **範例（C 語言）**  

```c
#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        printf("Hello from thread %d, i = %d\n",
               omp_get_thread_num(), i);
    }
    return 0;
}
```

編譯時加上 `-fopenmp`：

```bash
gcc -fopenmp example.c -o example
```

---  

### **2. OpenMPI**  

#### **什麼是 OpenMPI？**  

* OpenMPI (Open Message Passing Interface) 是一個 **分散式記憶體** 平行計算的標準實作。  
* 它主要用於 **多台機器（節點）** 組成的叢集 (cluster)，透過訊息傳遞來協調不同節點上的程序。  
* OpenMPI 提供了豐富的 API，用於進程間通訊、任務分配等。  

#### **特點**  

| 特點 | 說明 |
|------|------|
| 分散式記憶體 | 每個節點有自己的記憶體，需透過訊息傳遞交換資料。 |
| 多進程 | 程式在不同節點上以多個獨立進程執行。 |
| 可擴展性 | 支援數千乃至數萬個核心的大規模叢集。 |
| 複雜度 | 相較於 OpenMP，需要更多的程式設計與管理（例如手動分配任務、處理通訊）。 |

#### **適用場景**  

* 叢集運算：跨多台機器的平行計算。  
* 大規模科學模擬：氣象預報、材料科學、流體力學等。  
* 分散式資料處理：大數據分析、機器學習訓練等。  

#### **範例（C 語言）**  

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello from process %d/%d\n", rank, size);

    MPI_Finalize();
    return 0;
}
```

編譯與執行：

```bash
mpicc example.c -o example
mpirun -np 4 ./example   # 使用 4 個進程執行
```

---  

## **3. OpenMP vs OpenMPI：如何選擇？**  

| 比較項目 | OpenMP | OpenMPI |
|----------|--------|----------|
| 記憶體模型 | 共享記憶體 | 分散式記憶體 |
| 平行方式 | 多執行緒 | 多進程（訊息傳遞） |
| 適用硬體 | 單機多核心 | 多台機器組成的叢集 |
| 程式設計複雜度 | 簡單（加 pragma 即可） | 較複雜（需手動管理通訊） |
| 擴展性 | 有限（受限於單機核心數） | 高度擴展（可跨多機） |
| 效能 | 記憶體存取快，無需額外通訊 | 需要處理節點間通訊，開銷較大 |

### **選擇建議**  

* **單機多核心**：若程式主要在單一主機上執行，且想利用多核心，可使用 **OpenMP**。  
* **多機叢集**：若任務需要跨多台機器平行化，或要處理大規模資料，應使用 **OpenMPI**（或 MPI + OpenMP 混合模式）。  
* **混合模式**：在某些 HPC 應用中，會同時使用 **OpenMP**（單機內多執行緒） + **OpenMPI**（跨節點多進程），以達到最佳效能與擴展性。  

---  
