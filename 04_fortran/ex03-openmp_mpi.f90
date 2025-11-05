! hybrid.f90
program hybrid
    use mpi
    use omp_lib
    implicit none
    integer :: ierr, rank, size, thread_id, num_threads

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)

    !$omp parallel private(thread_id)
    thread_id = omp_get_thread_num()
    num_threads = omp_get_num_threads()
    write (*,*) "Process", rank, "out of", size, ": Hello from thread", thread_id, "out of", num_threads, "threads"
    !$omp end parallel

    call MPI_FINALIZE(ierr)
end program hybrid
