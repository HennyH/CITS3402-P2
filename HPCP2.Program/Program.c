// Program.c : This file contains the 'main' function. Program execution begins and ends there.

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define INT_INFINITY -1
#define INT_NULL -2

errno_t get_tile_i_j_for_process_rank(int rank, int n_vertices, int tile_dim, int* tile_i, int* tile_j)
{
  int n_tiles_in_axis = (int)ceil((double)n_vertices / (double)tile_dim);
  int curr_rank = 0;
  for (int i = 0; i < n_tiles_in_axis; i++) {
    for (int j = 0; j < n_tiles_in_axis; j++) {
      if (curr_rank++ == rank) {
        *tile_i = i;
        *tile_j = j;
        return 0;
      }
    }
  }

  return EINVAL;
}

errno_t apsp_floyd_warshall_distribute_tiles(int* adjacency_matrix, int n_vertices, int tile_dim, int root, MPI_Comm comm, int* tile_buffer)
{
  int n_processes, my_rank;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &n_processes);

  const int tile_size = (int)(tile_dim * tile_dim);
  int* tiles = NULL;

  if (my_rank == root) {
    tiles = calloc((size_t)n_processes * tile_size, sizeof(int));
    for (int i = 0; i < tile_size; i++) {
      tiles[i] = INT_NULL;
    }

    for (int rank = 0; rank < n_processes; rank++) {
      int send_tile_i, send_tile_j = 0;
      get_tile_i_j_for_process_rank(rank, n_vertices, tile_dim, &send_tile_i, &send_tile_j);
      const int from_vertex_start = send_tile_j * tile_dim;
      const int from_vertex_end = from_vertex_start + tile_dim;
      const int to_vertex_start = send_tile_i * tile_dim;
      const int to_vertex_end = to_vertex_start + tile_dim;
      int tile_value_i = 0;
      /* from vertex is the 'row' */
      for (int from_vertex = from_vertex_start; from_vertex < from_vertex_end; from_vertex++) {
        /* to vertex is the 'col' */
        for (int to_vertex = to_vertex_start; to_vertex < to_vertex_end; to_vertex++) {
          const int tile_i = rank * tile_size + (tile_value_i++);
          if (from_vertex >= n_vertices || to_vertex >= n_vertices) {
            tiles[tile_i] = INT_NULL;
          }
          else {
            const int adjacency_i = to_vertex + from_vertex * n_vertices;
            tiles[tile_i] = adjacency_matrix[adjacency_i];
          }
        }
      }
    }
  }

  MPI_Datatype tile_type;
  MPI_Type_vector(1, tile_size, tile_size, MPI_INT, &tile_type);
  MPI_Type_commit(&tile_type);
  MPI_Scatter(tiles, 1, tile_type, tile_buffer, tile_size, MPI_INT, root, comm);

  return 0;
}

errno_t apsp_floyd_warshall_create_tile_world_comm(int n_processes, MPI_Comm* tile_world_comm)
{
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  int* ranks = calloc(n_processes, sizeof(int));
  if (ranks == 0) {
    return ENOMEM;
  }

  for (int i = 0; i < n_processes; i++) {
    ranks[i] = i;
  }

  MPI_Group tile_group;
  MPI_Group_incl(world_group, n_processes, ranks, &tile_group);
  MPI_Comm_create(MPI_COMM_WORLD, tile_group, tile_world_comm);
  return 0;
}

errno_t apsp_floyd_warshall(int* adjacency_matrix, int n_vertices, int** results)
{
  int n_available_processes, my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &n_available_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* In the non-parallelized version we construct a adjacancy matrix `d` of size `n_vertices` x `n_vertices`. This adjacancy matrix
   * contains the upper bound of the distance between two particular verticies i and j in the cell located at (row=i, col=j). These estimates
   * are then 'relaxed' in a loop until they are optimal.
   *
   * We will parallelize this algorithim by subdiving `d` into 'tiles' which we can calculate in parallel. These tiles have dependencies on
   * data from other tiles - exactly what data they need will be detailed later on. We divide `d` up such that it consists of a tile for each
   * process we have.
   *
   * Math behind the calculation of tile_dim:
   *    processes × tile_dim²  = vertices²
   * => tile_dim²              = vertices² / processes
   * => tile_dim               = √(vertices² / processes)
   * => tile_dim               = vertices / √(processes)
   *
   * Note: The maximum parallelization would result in each tile being a single distance, you can't subdivide further than that...
   */
  const int max_parallelism = n_vertices * n_vertices;
  const int n_processes = min(max_parallelism, n_available_processes);
  /* If the process would have no work to do exit the function! */
  if (my_rank > n_processes) {
    return 0;
  }
  const int tile_dim = (int)ceil((double)n_vertices / sqrt(n_processes));
  const int tile_size = tile_dim * tile_dim;
  MPI_Comm tile_world_comm;
  apsp_floyd_warshall_create_tile_world_comm(n_processes, &tile_world_comm);

  int* tile = calloc(tile_size, sizeof(int));
  if (tile == NULL) {
    return ENOMEM;
  }
  apsp_floyd_warshall_distribute_tiles(adjacency_matrix, n_vertices, tile_dim, 0, tile_world_comm, tile);
  int tile_i, tile_j;
  get_tile_i_j_for_process_rank(my_rank, n_vertices, tile_dim, &tile_i, &tile_j);

#ifdef DEBUG
  if (my_rank == 0) {
    printf("apsp_floyd_warshall: divided %i veritices into %i tiles of dimension %i\n", n_vertices, n_processes, tile_dim);
  }
  printf("apsp_floyd_warshall: process %i processing tile (%i, %i): [%i, %i, %i, %i, %i, %i, %i, %i, %i]\n", my_rank, tile_i, tile_j, tile[0], tile[1], tile[2], tile[3], tile[4], tile[5], tile[6], tile[7], tile[8]);
#endif // DEBUG

  return 0;
}

void pause_to_allow_attachment()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    char answer[1];
    fprintf(stdout, "Press any key to continue...");
    fflush(stdout);
    fread_s(answer, sizeof(char) * _countof(answer), sizeof(char), _countof(answer), stdin);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
#ifdef DEBUG
  pause_to_allow_attachment();
#endif // DEBUG
  int adjacency_matrix[] = { 0, 15, 1, 1, 7,
                             0, 0,  3, 0, 7,
                             1, 3,  0, 0, 7,
                             0, 1,  1, 0, 7,
                             8, 9,  4, 3, 5 };
  apsp_floyd_warshall(adjacency_matrix, 5, NULL);
  MPI_Finalize();
  return 0;
}