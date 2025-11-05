# RISC-V64 Base HPC Clsuter 建置

以 RISC-V 64 架構硬體（DC ROMA L2A）之叢集計算環境(HPC)建置，包含 Scheduler、多核心/多執行緒之分散式計算建置與實做紀錄。

## 📋 索引

### 一、說明
- [專案概述](#-專案概述)
- [環境需求](#-環境需求)
- [元件補述](#-元件補述)

### 二、系統建置
- [Base OS](#base-os)
- [Scheduler](#scheduler)
- [Compiler](#compiler)

### 三、Lab:C++
- [OpenMP](#openmp-c)
- [MPI](#mpi-c)
- [Slurm](#slurm)
- [Matrix 矩陣運算](#matrix-矩陣運算)

### 三、Lab:Fortran
- [OpenMPI](#openmp-f90)
- [MPI](#mpi-f90)

---

# 一、說明

## 🎯 專案概述

本框架旨在使用 RISC-V64 架構硬體，進行叢集環境計算需求之環境建置，並透過實例進行驗證多核/多緒之於 RISC-V 分散式計算之可行性。

- **硬體**：
 - 🎯 DC-ROMA RISC-V LAPTOP II : https://deepcomputing.io/product/dc-roma-risc-v-laptop-ii/
 - HiFive Premier P550 : https://www.sifive.com/boards/hifive-premier-p550

* 環境建置流程示意
```
Base OS → Scheduler + Compiler + HPC Library → Labs
...
```

---

## 🛠️ 環境需求
- **實做環境** : DC-ROMA RISC-V LAPTOP II
  - **硬體設備**: DC-ROMA RISC-V LAPTOP II
  - **作業系統**: Ubuntu 24.04.3 LTS  
  - **調度排程**: Slurm
  - **身份認證**: Munge
  - **編譯器**: C/Clang/Fortran
  - **函式庫**: OpenMP/ OpenMPI

---

## 📖 元件補述
### **[ 📖 OpenMP v.s. OpenMPI](./note/openmpi-openmd.md)

# 二、系統建置

### Base OS
說明 OS 基礎環境與所需套件、網路配置、設定

### Scheduler
說明本實做之計算排程器之安裝與設定

### Compiler
說明本實做之編譯器與對應之函式庫安裝與設定

## 三、Lab:C++
以下為採用 C 語言之程式實做

### OpenMP (C)
C 實做 OpenMP 與測試

### MPI (C)
C 實做 OpenMPI 與測試

### Slurm
使用環境 Slurm (srun/sbatch) 進行 OpenMP/OpenMPI 測試

### Matrix 矩陣運算
以 C 實做矩陣生成與乘法運算，驗證多核多緒之分散式計算可行性

## 三、Lab:Fortran
以下為採用 Fortran 語言之程式實做

### OpenMP (F90)
Fortran 實做 OpenMP 與測試

### MPI (F90)
Fortran 實做 OpenMPI 與測試