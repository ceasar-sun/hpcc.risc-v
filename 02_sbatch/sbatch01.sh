#!/bin/bash
####SBATCH -A ENT108161 # Account name/project number
#SBATCH -J my_sbatch      # Job name
#SBATCH -p testpat
#SBATCH -n 12        # Number of MPI tasks (i.e. processes)
#SBATCH -c 1         # Number of cores per MPI task
#SBATCH -N 3         # Maximum number of nodes to be allocated
#SBATCH -t 05:00     # Wall time limit (days-hrs:min:sec)
#SBATCH -o /shared/workspace/02_sbatch/log/%u_%j.log   ## Path to the standard output and error files relative to the working directory

mpirun /shared/workspace/01_hello_mpi/hello_mpi

