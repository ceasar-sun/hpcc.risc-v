! ex02-mpi.f90
program mpi_hello
    use mpi
    implicit none
    integer :: ierr, rank, size

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
    write (*,*) "Hello from process", rank, "out of", size, "processes"
    call MPI_FINALIZE(ierr)
end program mpi_hello
