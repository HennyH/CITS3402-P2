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
/// Test if a tile contains a distance value for the given from_vertex -> to_vertex pair(s).
/// </summary>
/// <param name="from_vertex">From vertex (INT_NULL for wildcard).</param>
/// <param name="to_vertex">To vertex (INT_NULL for wildcard).</param>
/// <param name="tile_dim">The tile dimension.</param>
/// <param name="tile_i">The tile i.</param>
/// <param name="tile_j">The tile j.</param>
/// <returns>True if the tile contains the described pair(s).</returns>
inline int apsp_floyd_warshall_does_tile_cover_vertex_pair(int from_vertex, int to_vertex, int tile_dim, int tile_i, int tile_j)
{
  return (from_vertex == INT_NULL || (from_vertex >= tile_i * tile_dim && from_vertex < (tile_i + 1) * tile_dim))
    && (to_vertex == INT_NULL || (to_vertex >= tile_j * tile_dim && to_vertex < (tile_j + 1) * tile_dim));
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
errno_t apsp_floyd_warshall_tile_determine_i_j_for_process_rank(int rank, int n_vertices, int tile_dim, int* tile_i, int* tile_j)
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
      apsp_floyd_warshall_tile_determine_i_j_for_process_rank(rank, n_vertices, tile_dim, &send_tile_i, &send_tile_j);
      const int from_vertex_start = send_tile_i * tile_dim;
      const int from_vertex_end = from_vertex_start + tile_dim;
      const int to_vertex_start = send_tile_j * tile_dim;
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

  apsp_floyd_warshall_tile_determine_i_j_for_process_rank(my_rank, n_vertices, tile_dim, tile_i, tile_j);

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
      apsp_floyd_warshall_tile_determine_i_j_for_process_rank(rank, n_vertices, tile_dim, &tile_i, &tile_j);
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
    int status = MPI_Group_incl(tile_world_group, tile_matrix_dim, ranks_in_row, &tile_row_group);
    MPI_Comm_create(tile_world_comm, tile_row_group, tile_row_comms + row);
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
      apsp_floyd_warshall_tile_determine_i_j_for_process_rank(rank, n_vertices, tile_dim, &tile_i, &tile_j);
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
    MPI_Comm_create(tile_world_comm, tile_col_group, tile_col_comms + col);
  }

  return 0;
}

inline int apsp_floyd_warshall_does_tile_have_kth_col_segment(int k, int tile_dim, int tile_i, int tile_j)
{
  /* Our tile is contains the column if: tile_j * tile_dim <= k < (tile_j + 1) * tile_dim
   * Note: on the RHS we have '<' not a '<=' because we're 0 indexed.
   */
  return (tile_j * tile_dim <= k && k < tile_dim * (tile_j + 1));
}

inline int apsp_floyd_warshall_does_tile_have_kth_row_segment(int k, int tile_dim, int tile_i, int tile_j)
{
  /* Our tile is contains the row if: tile_i * tile_dim <= k < (tile_i + 1) * tile_dim
   * Note: on the RHS we have '<' not a '<=' because we're 0 indexed.
   */
  return (tile_i * tile_dim <= k && k < tile_dim * (tile_i + 1));
}

errno_t apsp_floyd_warshall_get_kth_col_segment(int k, int* tile, int tile_dim, int tile_i, int tile_j, int** col_segment)
{
  *col_segment = calloc(tile_dim, sizeof(int));

  if (!apsp_floyd_warshall_does_tile_have_kth_col_segment(k, tile_dim, tile_i, tile_j)) {
    return 0;
  }

  if (*col_segment == NULL) {
    return ENOMEM;
  }

  /* We're going to over each row in our tile and pluck out the value that occurs in the kth-col */
  const int kth_column_offset_within_tile = k - tile_j * tile_dim;
  for (int row = 0; row < tile_dim; row++) {
    (*col_segment)[row] = tile[row * tile_dim + kth_column_offset_within_tile];
  }

  return 0;
}

errno_t apsp_floyd_warshall_get_kth_row_segment(int k, int* tile, int tile_dim, int tile_i, int tile_j, int** row_segment)
{
  *row_segment = calloc(tile_dim, sizeof(int));

  if (!apsp_floyd_warshall_does_tile_have_kth_row_segment(k, tile_dim, tile_i, tile_j)) {
    return 0;
  }

  if (*row_segment == NULL) {
    return ENOMEM;
  }

  /* We're going to over each column in our tile and pluck out the value that occurs in the kth-row */
  const int kth_row_offset_within_tile = (k - tile_i * tile_dim) * tile_dim;
  for (int col = 0; col < tile_dim; col++) {
    (*row_segment)[col] = tile[kth_row_offset_within_tile + col];
  }

  return 0;
}

errno_t apsp_floyd_warshall_determine_tile_i_j_which_covers_kth_row_and_col(int k, int tile_matrix_dim, int tile_dim, int* tile_i, int* tile_j)
{
  for (int i = 0; i < tile_matrix_dim; i++) {
    for (int j = 0; j < tile_matrix_dim; j++) {
      if (apsp_floyd_warshall_does_tile_cover_vertex_pair(k, k, tile_dim, i, j)) {
        *tile_i = i;
        *tile_j = j;
        return 0;
      }
    }
  }

  return EINVAL;
}

errno_t apsp_floyd_warshall_recieve_required_kth_row_and_column_segments(int k, int my_rank, int n_vertices, int tile_matrix_dim, int* tile, int tile_dim, int tile_i, int tile_j, MPI_Comm tile_world_comm, MPI_Comm* tile_row_comms, MPI_Comm* tile_col_comms, int** required_kth_row_segment, int** required_kth_col_segment)
{
  MPI_Datatype segment_type;
  MPI_Type_contiguous(tile_dim, MPI_INT, &segment_type);
  MPI_Type_commit(&segment_type);

  int kth_col_and_row_tile_i, kth_col_and_row_tile_j;
  apsp_floyd_warshall_determine_tile_i_j_which_covers_kth_row_and_col(k, tile_matrix_dim, tile_dim, &kth_col_and_row_tile_i, &kth_col_and_row_tile_j);

  if (tile_i == kth_col_and_row_tile_i) {
    apsp_floyd_warshall_get_kth_row_segment(k, tile, tile_dim, tile_i, tile_j, required_kth_row_segment);
  }
  else {
    *required_kth_row_segment = calloc(tile_dim, sizeof(int));
  }

  if (tile_j == kth_col_and_row_tile_j) {
    apsp_floyd_warshall_get_kth_col_segment(k, tile, tile_dim, tile_i, tile_j, required_kth_col_segment);
  }
  else {
    *required_kth_col_segment = calloc(tile_dim, sizeof(int));
  }

  printf("process %i is going to give out col = [%i, %i] and row = [%i, %i]\n", my_rank, (*required_kth_col_segment)[0], (*required_kth_col_segment)[1], (*required_kth_row_segment)[0], (*required_kth_row_segment)[1]);

  MPI_Bcast(*required_kth_row_segment, 1, segment_type, kth_col_and_row_tile_i, tile_col_comms[tile_j]);
  MPI_Bcast(*required_kth_col_segment, 1, segment_type, kth_col_and_row_tile_j, tile_row_comms[tile_i]);
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
    int* kth_col_segment;
    int* kth_row_segment;
    apsp_floyd_warshall_recieve_required_kth_row_and_column_segments(k, my_rank, n_vertices, tile_matrix_dim, tile, tile_dim, tile_i, tile_j, tile_world_comm, tile_row_comms, tile_col_comms, &kth_row_segment, &kth_col_segment);
    const int from_vertex_start = tile_i * tile_dim;
    const int from_vertex_end = from_vertex_start + tile_dim;
    const int to_vertex_start = tile_j * tile_dim;
    const int to_vertex_end = to_vertex_start + tile_dim;
    int tile_value_i = 0;
#ifdef DEBUG
    printf("apsp_floyd_warshall: process %i on iteration k = %i recieved col = [%i, %i] and row = [%i, %i]\n", my_rank, k, kth_col_segment[0], kth_col_segment[1], kth_row_segment[0], kth_row_segment[1]);
#endif // DEBUG
    /* from vertex is the 'row' */
    for (int from_vertex = from_vertex_start; from_vertex < from_vertex_end; from_vertex++) {
      /* to vertex is the 'col' */
      for (int to_vertex = to_vertex_start; to_vertex < to_vertex_end; to_vertex++) {
        tile[(from_vertex * tile_dim) + to_vertex] = min(tile[(from_vertex * tile_dim) + to_vertex], kth_col_segment[from_vertex - from_vertex_start] + kth_row_segment[to_vertex - to_vertex_start]);
      }
    }
    printf("apsp_floyd_warshall: process %i on iteration k = %i computed tile = [%i, %i, %i, %i]\n", my_rank, k, tile[0], tile[1], tile[2], tile[3]);
  }

  printf("apsp_floyd_warshall: process %i is finished tile (%i, %i) = [%i, %i, %i, %i]\n", my_rank, tile_i, tile_j, tile[0], tile[1], tile[2], tile[3]);
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
  int adjacency_matrix[] = { 0, 9999, -2, 9999,
                             4, 0,     3, 9999,
                             9999, 9999,  0, 2,
                             9999, -1,  9999, 0 };
  apsp_floyd_warshall(adjacency_matrix, 4, NULL);
  MPI_Finalize();
  return 0;
}