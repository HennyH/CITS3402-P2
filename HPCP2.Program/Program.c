// Program.c : This file contains the 'main' function. Program execution begins and ends there.

#include <mpi.h>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Finalize();
  return 0;
}