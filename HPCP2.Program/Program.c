// Program.c : This file contains the 'main' function. Program execution begins and ends there.

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define INT_INFINITY -1
#define INT_NULL -2

/// <summary>
/// Calculates the dimension of the tiles which compose the 'tile matrix'.
/// </summary>
/// <param name="n_vertices">The n vertices.</param>
/// <param name="n_processes">The n processes.</param>
/// <returns>The dimension of the tiles.</returns>
inline int apsp_floyd_warshall_calc_tile_dim(int n_vertices, int n_processes)
{
  /*
  * Math behind the calculation of tile_dim:
  *    processes × tile_dim²  = vertices²
  * => tile_dim²              = vertices² / processes
  * => tile_dim               = √(vertices² / processes)
  * => tile_dim               = vertices / √(processes)
  */
  return (int)ceil((double)n_vertices / sqrt(n_processes));
}

/// <summary>
/// Calculates the dimension of the 'tile matrix' (the matrix of tiles which overlays the adjacency matrix).
/// </summary>
/// <param name="n_vertices">The n vertices.</param>
/// <param name="n_processes">The n processes.</param>
/// <returns>The dimension of the tile matrix.</returns>
inline int apsp_floyd_warshall_calc_tile_matrix_dim(int n_vertices, int tile_dim)
{
  return (int)ceil((double)n_vertices / (double)tile_dim);
}

/// <summary>
/// Determine the tile position (row=tile_i, col=tile_j) which a process with the given rank is responsible for processing.
/// </summary>
/// <param name="rank">The rank.</param>
/// <param name="n_vertices">The n vertices.</param>
/// <param name="tile_dim">The dimension of the tiles.</param>
/// <param name="tile_i">out: The tile_i.</param>
/// <param name="tile_j">out: The tile_j.</param>
/// <returns>An error code.</returns>
errno_t apsp_floyd_warshall_tile_i_j_for_process_rank(int rank, int n_vertices, int tile_dim, int* tile_i, int* tile_j)
{
  int n_tiles_in_axis = apsp_floyd_warshall_calc_tile_matrix_dim(n_vertices, tile_dim);
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

/// <summary>
/// Partitions an adjacency matrix into appropriatley sized tiles and scatters them to processes in the given communicator.
/// </summary>
/// <param name="adjacency_matrix">The adjacency matrix.</param>
/// <param name="n_vertices">The n vertices.</param>
/// <param name="tile_dim">The dimension of the tiles.</param>
/// <param name="root">The root process which sends the data.</param>
/// <param name="comm">The communicator.</param>
/// <param name="tile_i">out: The recieved tile_i.</param>
/// <param name="tile_j">out: The recieved tile_j.</param>
/// <param name="tile_buffer">inout: The buffer to store the recieved tile.</param>
/// <returns>An error code.</returns>
errno_t apsp_floyd_warshall_distribute_tiles(int* adjacency_matrix, int n_vertices, int tile_dim, int root, MPI_Comm comm, int* tile_i, int* tile_j, int* tile_buffer)
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
      apsp_floyd_warshall_tile_i_j_for_process_rank(rank, n_vertices, tile_dim, &send_tile_i, &send_tile_j);
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

  apsp_floyd_warshall_tile_i_j_for_process_rank(my_rank, n_vertices, tile_dim, tile_i, tile_j);

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

errno_t apsp_floyd_warshall_create_row_comms(MPI_Comm tile_world_comm, int n_vertices, int tile_dim, int tile_matrix_dim, MPI_Comm* const tile_row_comms)
{
  int n_tiles;
  MPI_Comm_size(tile_world_comm, &n_tiles);

  MPI_Group tile_world_group;
  MPI_Comm_group(tile_world_comm, &tile_world_group);

  for (int row = 0; row < tile_matrix_dim; row++) {
    /* This array will contain the _ranks_ of the processes (within the tile world) which are processing a tile along the current row. */
    int* ranks_in_row = calloc(tile_matrix_dim, sizeof(int));
    if (ranks_in_row == NULL) {
      return ENOMEM;
    }
    for (int rank = 0; rank < n_tiles; rank++) {
      int tile_i, tile_j;
      apsp_floyd_warshall_tile_i_j_for_process_rank(rank, n_vertices, tile_dim, &tile_i, &tile_j);
      /* This partiuclar rank corresponds to the tile at (tile_i, tile_j) and so we check if that tile is along the current row. We
       * can use the `tile_j` as the index to store this rank at because by definition they go from 0...tile_matrix_dim (which is the size
       * of the matrix we allocated).
       */
      if (tile_i == row) {
        ranks_in_row[tile_j] = rank;
      }
    }

    /* Now that we know the ranks of the tiles which occur in the row we create a communicator which includes them all derived from tile world. */
    MPI_Group tile_row_group;
    MPI_Group_incl(tile_world_group, tile_matrix_dim, ranks_in_row, &tile_row_group);
    tile_row_comms[row] = tile_row_group;
  }

  return 0;
}

errno_t apsp_floyd_warshall_create_col_comms(MPI_Comm tile_world_comm, int n_vertices, int tile_dim, int tile_matrix_dim, MPI_Comm* const tile_col_comms)
{
  int n_tiles;
  MPI_Comm_size(tile_world_comm, &n_tiles);

  MPI_Group tile_world_group;
  MPI_Comm_group(tile_world_comm, &tile_world_group);

  for (int col = 0; col < tile_matrix_dim; col++) {
    /* This array will contain the _ranks_ of the processes (within the tile world) which are processing a tile along the current col. */
    int* ranks_in_col = calloc(tile_matrix_dim, sizeof(int));
    if (ranks_in_col == NULL) {
      return ENOMEM;
    }
    for (int rank = 0; rank < n_tiles; rank++) {
      int tile_i, tile_j;
      apsp_floyd_warshall_tile_i_j_for_process_rank(rank, n_vertices, tile_dim, &tile_i, &tile_j);
      /* This partiuclar rank corresponds to the tile at (tile_i, tile_j) and so we check if that tile is along the current col. We
       * can use the `tile_i` as the index to store this rank at because by definition they go from 0...tile_matrix_dim (which is the size
       * of the matrix we allocated).
       */
      if (tile_j == col) {
        ranks_in_col[tile_i] = rank;
      }
    }

    /* Now that we know the ranks of the tiles which occur in the col we create a communicator which includes them all derived from tile world. */
    MPI_Group tile_col_group;
    MPI_Group_incl(tile_world_group, tile_matrix_dim, ranks_in_col, &tile_col_group);
    tile_col_comms[col] = tile_col_group;
  }

  return 0;
}

errno_t apsp_floyd_warshall_get_kth_col_segment(int k, int* tile, int tile_dim, int tile_i, int tile_j, int** col_segment)
{
  /* quickly check that the kth column actually goes through our tile! If it doesn't we should set the col_segment pointer to NULL to let the
   * caller know.
   *
   * Our tile is contains the column if: tile_i * tile_dim <= k < (tile_i + 1) * tile_dim
   * Note: on the RHS we have '<' not a '<=' because we're 0 indexed.
   */
  if (!(tile_i * tile_dim <= k && k < tile_dim * (tile_i + 1))) {
    *col_segment = NULL;
    return 0;
  }

  *col_segment = calloc(tile_dim, sizeof(int));
  if (*col_segment == NULL) {
    return ENOMEM;
  }

  /* We're going to over each row in our tile and pluck out the value that occurs in the kth-col */
  const int kth_column_offset_within_tile = k - tile_i * tile_dim;
  for (int row = 0; row < tile_dim; row++) {
    (*col_segment)[row] = tile[row * tile_dim + kth_column_offset_within_tile];
  }

  return 0;
}

errno_t apsp_floyd_warshall_get_kth_row_segment(int k, int* tile, int tile_dim, int tile_i, int tile_j, int** row_segment)
{
  /* quickly check that the kth row actually goes through our tile! If it doesn't we should set the row_segment pointer to NULL to let the
   * caller know.
   *
   * Our tile is contains the row if: tile_j * tile_dim <= k < (tile_j + 1) * tile_dim
   * Note: on the RHS we have '<' not a '<=' because we're 0 indexed.
   */
  if (!(tile_j * tile_dim <= k && k < tile_dim * (tile_j + 1))) {
    *row_segment = NULL;
    return 0;
  }

  *row_segment = calloc(tile_dim, sizeof(int));
  if (*row_segment == NULL) {
    return ENOMEM;
  }

  /* We're going to over each column in our tile and pluck out the value that occurs in the kth-row */
  const int kth_row_offset_within_tile = (k - tile_j * tile_dim) * tile_dim;
  for (int col = 0; col < tile_dim; col++) {
    (*row_segment)[col] = tile[kth_row_offset_within_tile + col];
  }

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
   * Note: The maximum parallelization would result in each tile being a single distance, you can't subdivide further than that...
   */
  const int max_parallelism = n_vertices * n_vertices;
  const int n_processes = min(max_parallelism, n_available_processes);
  /* If the process would have no work to do exit the function! */
  if (my_rank > n_processes) {
    return 0;
  }

  const int tile_dim = apsp_floyd_warshall_calc_tile_dim(n_vertices, n_processes);
  const int tile_matrix_dim = apsp_floyd_warshall_calc_tile_matrix_dim(n_vertices, tile_dim);
  const int tile_size = tile_dim * tile_dim;
  int* const tile = calloc(tile_size, sizeof(int));
  int tile_i, tile_j;
  if (tile == NULL) {
    return ENOMEM;
  }

  /* Create the various communicators that facilitate the distribution of messages to particular sections of the tiled adjacancy matrix:
   * - tile_world_comm: This is a communicator which includes each process that will be given a tile to process.
   * - tile_row_comm[]: A collection of communicators such that the i-th entry includes all processes processing a tile along the i-th tile row.
   * - tile_col_comm[]: A collection of communicators such that the j-th entry includes all processes processing a tile along the j-th tile col.
   */
  MPI_Comm tile_world_comm;
  MPI_Comm* const tile_row_comms = calloc(tile_matrix_dim, sizeof(MPI_Comm));
  MPI_Comm* const tile_col_comms = calloc(tile_matrix_dim, sizeof(MPI_Comm));
  apsp_floyd_warshall_create_tile_world_comm(n_processes, &tile_world_comm);
  MPI_Comm_rank(tile_world_comm, &my_rank);
  apsp_floyd_warshall_create_row_comms(tile_world_comm, n_vertices, tile_dim, tile_matrix_dim, tile_row_comms);
  apsp_floyd_warshall_create_col_comms(tile_world_comm, n_vertices, tile_dim, tile_matrix_dim, tile_col_comms);

  /* This is the initial communication stage which distributes the tiles to each process. Underneath this method uses a collective
   * scatter communication and hence all processes should call the method.
   */
  apsp_floyd_warshall_distribute_tiles(adjacency_matrix, n_vertices, tile_dim, 0, tile_world_comm, &tile_i, &tile_j, tile);

#ifdef DEBUG
  if (my_rank == 0) {
    printf("apsp_floyd_warshall: scattered the %i dimension displacement matrix into %i tiles of dimension %i\n", n_vertices, n_processes, tile_dim);
  }
  printf("apsp_floyd_warshall: process %i is processing tile (%i, %i) = [%i, %i, %i, %i]\n", my_rank, tile_i, tile_j, tile[0], tile[1], tile[2], tile[3]);
#endif // DEBUG

  for (int k = 0; k < n_vertices; k++) {
    int* my_kth_row_segment = NULL;
    int* my_kth_col_segment = NULL;
    apsp_floyd_warshall_get_kth_row_segment(k, tile, tile_dim, tile_i, tile_j, &my_kth_row_segment);
    apsp_floyd_warshall_get_kth_col_segment(k, tile, tile_dim, tile_i, tile_j, &my_kth_col_segment);
#ifdef DEBUG
    printf("apsp_floyd_warshall: process %i processing tile (%i, %i) for k = %i ", my_rank, tile_i, tile_j, k);
    if (my_kth_col_segment == NULL) {
      printf("has no column segment ");
    }
    else {
      printf("has a column segment of [%i, %i] ", my_kth_col_segment[0], my_kth_col_segment[1]);
    }
    if (my_kth_row_segment == NULL) {
      printf("and has no row segment ");
    }
    else {
      printf("and has a row segment of [%i, %i] ", my_kth_row_segment[0], my_kth_row_segment[1]);
    }
    printf("\n");
#endif // DEBUG
  }
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
  int adjacency_matrix[] = { 0, 15, 1, 1,
                             0, 0,  3, 0,
                             1, 3,  0, 0,
                             0, 1,  1, 0, };
  apsp_floyd_warshall(adjacency_matrix, 4, NULL);
  MPI_Finalize();
  return 0;
}