/*=================================================================
 * openp_ex01.c
 *   - 顯示可用的 OpenMP 執行緒數 (＝ 可見的 CPU 核心數)
 *   - 以 parallel for 印出每個迭代所屬的執行緒
 *=================================================================*/

#include <stdio.h>
#include <omp.h>

int main(void)
{
    /* -------------------------------------------------------------
     * 1️⃣ 取得「預設」的執行緒上限
     *    - 預設值＝系統可見的邏輯 CPU 數 (通常等於 core 數*超執行緒)
     *    - 若環境變數 OMP_NUM_THREADS 已設定，會以該值為準
     * ------------------------------------------------------------ */
    int max_threads = omp_get_max_threads();
    printf("=== OpenMP 可用執行緒數 (max_threads) = %d ===\n", max_threads);

    int num_procs = omp_get_num_procs();
    printf("=== OpenMP 系統可見的邏輯 CPU 數 (num_procs) = %d ===\n", num_procs);

    /* -------------------------------------------------------------
     * 2️⃣ 以 parallel for 產生平行迴圈
     * ------------------------------------------------------------ */
    #pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        /* 每次迭代都列印本執行緒的 ID 與迭代變數 i */
        printf("Hello from thread %d, i = %d\n",
               omp_get_thread_num(), i);
    }

    return 0;
}

