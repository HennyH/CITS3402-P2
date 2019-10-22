// Program.c : This file contains the 'main' function. Program execution begins and ends there.

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <stdarg.h>

#define INT_INFINITY 9999
#define INT_NULL -2

/// <summary>
/// Calculates the dimension of the 'tile matrix' (the matrix of tiles which overlays the adjacency matrix).
/// </summary>
/// <param name="n_vertices">The n vertices.</param>
/// <param name="n_available_processes">The number of processes available.</param>
/// <returns>The dimension of the tiles.</returns>
inline int apsp_floyd_warshall_calc_tile_matrix_dim(int n_vertices, int n_available_processes)
{
  return (int)floor(sqrt(min(n_vertices, n_available_processes)));
}

/// <summary>
/// Calculates the dimension of the tiles which compose the 'tile matrix'.
/// </summary>
/// <param name="n_vertices">The n vertices.</param>
/// <param name="n_processes">The n processes.</param>
/// <returns>The dimension of the tile matrix.</returns>
inline int apsp_floyd_warshall_calc_tile_dim(int n_vertices, int time_matrix_dim)
{
  return (int)ceil((double)n_vertices / (double)time_matrix_dim);
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
errno_t apsp_floyd_warshall_tile_determine_i_j_for_process_rank(int rank, int tile_matrix_dim, int* tile_i, int* tile_j)
{
  int curr_rank = 0;
  for (int i = 0; i < tile_matrix_dim; i++) {
    for (int j = 0; j < tile_matrix_dim; j++) {
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
errno_t apsp_floyd_warshall_distribute_tiles(int* adjacency_matrix, int n_vertices, int tile_dim, int tile_matrix_dim, int root, MPI_Comm comm, MPI_Datatype tile_type, int* tile_i, int* tile_j, int* tile_buffer)
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
      apsp_floyd_warshall_tile_determine_i_j_for_process_rank(rank, tile_matrix_dim, &send_tile_i, &send_tile_j);
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

  MPI_Scatter(tiles, 1, tile_type, tile_buffer, tile_size, MPI_INT, root, comm);

  if (my_rank == root && tiles != NULL) {
    free(tiles);
  }

  apsp_floyd_warshall_tile_determine_i_j_for_process_rank(my_rank, tile_matrix_dim, tile_i, tile_j);

  return 0;
}

/// <summary>
/// Determine the domain of vertices the tile covers. That is what `from_vertex` -> `to_vertex` pairs does the tile include.
/// </summary>
/// <param name="tile_dim">The tile dim.</param>
/// <param name="tile_i">The tile i.</param>
/// <param name="tile_j">The tile j.</param>
/// <param name="from_vertex_start">From vertex start.</param>
/// <param name="from_vertex_end">From vertex end.</param>
/// <param name="to_vertex_start">To vertex start.</param>
/// <param name="to_vertex_end">To vertex end.</param>
/// <returns>The from and to vertex domain start and end pairs.</returns>
errno_t apsp_floyd_warshall_determine_tile_vertex_domain(int n_vertices, int tile_dim, int tile_i, int tile_j, int* from_vertex_start, int* from_vertex_end, int* to_vertex_start, int* to_vertex_end)
{
  *from_vertex_start = tile_i * tile_dim;
  *from_vertex_end = min(*from_vertex_start + tile_dim, n_vertices);
  *to_vertex_start = tile_j * tile_dim;
  *to_vertex_end = min(*to_vertex_start + tile_dim, n_vertices);
  return 0;
}

errno_t apsp_floyd_warshall_gather_tiles_to_root(int* adjacency_matrix, int n_vertices, int tile_dim, int tile_matrix_dim, int root, int* tile, MPI_Comm comm)
{
  int n_processes, my_rank;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &n_processes);

  /* You may be  wondering why `n_values` isn't n_verticies² - the reason is that for ceartin sized adjacency matrices we can't
   * evenly distribute the matrix between the tiles, this results in some processes having 'haloed' (INT_NULL) values in their tile. These
   * 'null' values are there just so that each process has the same _sized_ tile. Because all the operations iterate over vertex pairs we
   * never end up touching these null values.
   */
  const int n_tiles = tile_matrix_dim * tile_matrix_dim;
  const int n_values = n_tiles * (tile_dim * tile_dim);
  int* values = NULL;
  if (my_rank == root) {
    values = calloc(n_values, sizeof(int));
  }

  const int n_values_in_tile = tile_dim * tile_dim;
  MPI_Gather(tile, n_values_in_tile, MPI_INT, values, n_values_in_tile, MPI_INT, root, comm);

  if (my_rank != root) {
    return 0;
  }

  if (values == NULL) {
    return ENOMEM;
  }

  /* Now that we have all the tiles sitting in memory (possibly with INT_NULL values!) we need to update our adjacecency matrix with the final results.
   * We can't just magic-copy them in because the tiles don't align nicely with contigous segments of memory. And also because of the INT_NULL values the
   * use of a MPI vector type + Gatherv won't work hence we need to do a little work here. The idea is we know every (from_vertex, to_vertex) pair and so
   * just go through each one, for each pair we know the tile that includes it, we then look into that tile.
   */
  for (int rank = 0; rank < n_tiles; rank++) {
    int* tile = (values + ((size_t)rank * (size_t)n_values_in_tile));
    int tile_i, tile_j;
    apsp_floyd_warshall_tile_determine_i_j_for_process_rank(rank, tile_matrix_dim, &tile_i, &tile_j);
    int from_vertex_start, from_vertex_end, to_vertex_start, to_vertex_end;
    apsp_floyd_warshall_determine_tile_vertex_domain(n_vertices, tile_dim, tile_i, tile_j, &from_vertex_start, &from_vertex_end, &to_vertex_start, &to_vertex_end);
    /* from vertex is the 'row' */
    for (int from_vertex = from_vertex_start; from_vertex < from_vertex_end; from_vertex++) {
      /* to vertex is the 'col' */
      for (int to_vertex = to_vertex_start; to_vertex < to_vertex_end; to_vertex++) {
        const int d_ij_tile_index = ((from_vertex - from_vertex_start) * tile_dim) + (to_vertex - to_vertex_start);
        const int d_ij_index = (from_vertex * n_vertices) + to_vertex;
        adjacency_matrix[d_ij_index] = tile[d_ij_tile_index];
      }
    }
  }

  free(values);
  return 0;
}

errno_t apsp_floyd_warshall_create_tile_world_comm(int n_processes, MPI_Comm* tile_world_comm)
{
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  int* ranks = calloc(n_processes, sizeof(int));
  if (ranks == NULL) {
    return ENOMEM;
  }

  for (int i = 0; i < n_processes; i++) {
    ranks[i] = i;
  }

  MPI_Group tile_group;
  MPI_Group_incl(world_group, n_processes, ranks, &tile_group);
  MPI_Comm_create(MPI_COMM_WORLD, tile_group, tile_world_comm);
  free(ranks);
  if (*tile_world_comm == MPI_COMM_NULL) {
    fprintf(stderr, "apsp_floyd_warshall_create_tile_world_comm: failed to create tile comm world.");
    return EINVAL;
  }

  return 0;
}

errno_t apsp_floyd_warshall_create_row_comms(MPI_Comm tile_world_comm, int n_vertices, int tile_dim, int tile_matrix_dim, MPI_Comm* const tile_row_comms)
{
  int n_tiles, my_rank;
  MPI_Comm_size(tile_world_comm, &n_tiles);
  MPI_Comm_rank(tile_world_comm, &my_rank);

  MPI_Group tile_world_group;
  MPI_Comm_group(tile_world_comm, &tile_world_group);

  /* This array will contain the _ranks_ of the processes (within the tile world) which are processing a tile along the current row. */
  int* ranks_in_row = calloc(tile_matrix_dim, sizeof(int));
  if (ranks_in_row == NULL) {
    return ENOMEM;
  }

  for (int row = 0; row < tile_matrix_dim; row++) {
    int i_am_in_row = 0;
    for (int rank = 0; rank < n_tiles; rank++) {
      int tile_i, tile_j;
      apsp_floyd_warshall_tile_determine_i_j_for_process_rank(rank, tile_matrix_dim, &tile_i, &tile_j);
      /* This partiuclar rank corresponds to the tile at (tile_i, tile_j) and so we check if that tile is along the current row. We
       * can use the `tile_j` as the index to store this rank at because by definition they go from 0...tile_matrix_dim (which is the size
       * of the matrix we allocated).
       */
      if (tile_i == row) {
        ranks_in_row[tile_j] = rank;
        i_am_in_row |= rank == my_rank;
      }
    }

    /* Now that we know the ranks of the tiles which occur in the row we create a communicator which includes them all derived from tile world. */
    MPI_Group tile_row_group;
    int status = MPI_Group_incl(tile_world_group, tile_matrix_dim, ranks_in_row, &tile_row_group);
    MPI_Comm* tile_row_comm = tile_row_comms + row;
    status = MPI_Comm_create(tile_world_comm, tile_row_group, tile_row_comm);
    if (i_am_in_row && *tile_row_comm == MPI_COMM_NULL) {
      fprintf(stderr, "apsp_floyd_warshall_create_row_comms: process %i failed to create row comm world for row %i.\n", my_rank, row);
      return EINVAL;
    }
  }

  free(ranks_in_row);
  return 0;
}

errno_t apsp_floyd_warshall_create_col_comms(MPI_Comm tile_world_comm, int n_vertices, int tile_dim, int tile_matrix_dim, MPI_Comm* const tile_col_comms)
{
  int n_tiles, my_rank;
  MPI_Comm_size(tile_world_comm, &n_tiles);
  MPI_Comm_rank(tile_world_comm, &my_rank);

  MPI_Group tile_world_group;
  MPI_Comm_group(tile_world_comm, &tile_world_group);

  /* This array will contain the _ranks_ of the processes (within the tile world) which are processing a tile along the current col. */
  int* ranks_in_col = calloc(tile_matrix_dim, sizeof(int));
  if (ranks_in_col == NULL) {
    return ENOMEM;
  }
  for (int col = 0; col < tile_matrix_dim; col++) {
    int i_am_in_col = 0;
    for (int rank = 0; rank < n_tiles; rank++) {
      int tile_i, tile_j;
      apsp_floyd_warshall_tile_determine_i_j_for_process_rank(rank, tile_matrix_dim, &tile_i, &tile_j);
      /* This partiuclar rank corresponds to the tile at (tile_i, tile_j) and so we check if that tile is along the current col. We
       * can use the `tile_i` as the index to store this rank at because by definition they go from 0...tile_matrix_dim (which is the size
       * of the matrix we allocated).
       */
      if (tile_j == col) {
        ranks_in_col[tile_i] = rank;
        i_am_in_col |= rank == my_rank;
      }
    }

    /* Now that we know the ranks of the tiles which occur in the col we create a communicator which includes them all derived from tile world. */
    MPI_Group tile_col_group;
    MPI_Group_incl(tile_world_group, tile_matrix_dim, ranks_in_col, &tile_col_group);
    MPI_Comm* tile_col_comm = tile_col_comms + col;
    MPI_Comm_create(tile_world_comm, tile_col_group, tile_col_comm);
    if (i_am_in_col && *tile_col_comm == MPI_COMM_NULL) {
      fprintf(stderr, "apsp_floyd_warshall_create_col_comms: failed to create col comm world for col %i.\n", col);
      return EINVAL;
    }
  }

  free(ranks_in_col);
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
  if (!apsp_floyd_warshall_does_tile_have_kth_col_segment(k, tile_dim, tile_i, tile_j)) {
    for (int i = 0; i < tile_dim; i++) {
      (*col_segment)[i] = 0;
    }
  }
  else {
    /* We're going to over each row in our tile and pluck out the value that occurs in the kth-col */
    const int kth_column_offset_within_tile = k - tile_j * tile_dim;
    for (int row = 0; row < tile_dim; row++) {
      (*col_segment)[row] = tile[row * tile_dim + kth_column_offset_within_tile];
    }
  }

  return 0;
}

errno_t apsp_floyd_warshall_get_kth_row_segment(int k, int* tile, int tile_dim, int tile_i, int tile_j, int** row_segment)
{
  if (!apsp_floyd_warshall_does_tile_have_kth_row_segment(k, tile_dim, tile_i, tile_j)) {
    for (int i = 0; i < tile_dim; i++) {
      (*row_segment)[i] = 0;
    }
  }
  else {
    /* We're going to over each column in our tile and pluck out the value that occurs in the kth-row */
    const int kth_row_offset_within_tile = (k - tile_i * tile_dim) * tile_dim;
    for (int col = 0; col < tile_dim; col++) {
      (*row_segment)[col] = tile[kth_row_offset_within_tile + col];
    }
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

errno_t apsp_floyd_warshall_recieve_required_kth_row_and_column_segments(int k, int my_rank, int n_vertices, int tile_matrix_dim, int* tile, int tile_dim, int tile_i, int tile_j, MPI_Comm tile_world_comm, MPI_Comm* tile_row_comms, MPI_Comm* tile_col_comms, MPI_Datatype segment_type, int** required_kth_row_segment, int** required_kth_col_segment)
{
  int kth_col_and_row_tile_i, kth_col_and_row_tile_j;
  apsp_floyd_warshall_determine_tile_i_j_which_covers_kth_row_and_col(k, tile_matrix_dim, tile_dim, &kth_col_and_row_tile_i, &kth_col_and_row_tile_j);

  apsp_floyd_warshall_get_kth_row_segment(k, tile, tile_dim, tile_i, tile_j, required_kth_row_segment);
  apsp_floyd_warshall_get_kth_col_segment(k, tile, tile_dim, tile_i, tile_j, required_kth_col_segment);

  MPI_Bcast(*required_kth_row_segment, 1, segment_type, kth_col_and_row_tile_i, tile_col_comms[tile_j]);
  MPI_Bcast(*required_kth_col_segment, 1, segment_type, kth_col_and_row_tile_j, tile_row_comms[tile_i]);
  return 0;
}

void debug_print_matrix(FILE* file, char* const message, int* values, int rows, int cols, ...)
{
  va_list fmt_args;
  va_start(fmt_args, cols);

  vfprintf(file, message, fmt_args);
  fprintf(file, "\n\t");
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      const int value = values[row * cols + col];
      if (value >= INT_INFINITY) {
        fprintf(file, " %5s", "(><)");
      }
      else {
        fprintf(file, " %5i", value);
      }
    }
    if (row == rows - 1) {
      fprintf(file, "\n");
    }
    else {
      fprintf(file, "\n\t");
    }
  }
}

errno_t apsp_floyd_warshall(int* adjacency_matrix, int n_vertices)
{
  int n_available_processes, my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &n_available_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Only the root process reads the file but all processes need the number of verticies during the setup portion.
   * Each process recieves their portion of the adjacency matrix later.
   */
  MPI_Bcast(&n_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
  const int tile_matrix_dim = apsp_floyd_warshall_calc_tile_matrix_dim(n_vertices, n_available_processes);
  const int tile_dim = apsp_floyd_warshall_calc_tile_dim(n_vertices, tile_matrix_dim);
  const int n_processes = tile_matrix_dim * tile_matrix_dim;
#ifdef DEBUG
  if (my_rank == 0) {
    fprintf(stderr, "n_verts=%i, tile_dim=%i, tile_matrix_dim=%i, n_aval_processes=%i, n_processes=%i\n", n_vertices, tile_dim, tile_matrix_dim, n_available_processes, n_processes);
  }
#endif // DEBUG

  /* If the process would have no work to do exit the function! Remember that our ranks are 0 indexed hence the >=! */
  if (my_rank >= n_processes) {
#ifdef DEBUG
    fprintf(stderr, "apsp_floyd_warshall: process %i exiting and not participating in calculation\n", my_rank);
#endif // DEBUG
    return 0;
  }

#ifdef DEBUG
  fprintf(stderr, "apsp_floyd_warshall: process %i participating in calculation\n", my_rank);
#endif // DEBUG

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
  apsp_floyd_warshall_create_row_comms(tile_world_comm, n_vertices, tile_dim, tile_matrix_dim, tile_row_comms);
  apsp_floyd_warshall_create_col_comms(tile_world_comm, n_vertices, tile_dim, tile_matrix_dim, tile_col_comms);
  MPI_Comm_rank(tile_world_comm, &my_rank);

  MPI_Datatype tile_type;
  MPI_Type_vector(1, tile_size, tile_size, MPI_INT, &tile_type);
  MPI_Type_commit(&tile_type);

  MPI_Datatype segment_type;
  MPI_Type_contiguous(tile_dim, MPI_INT, &segment_type);
  MPI_Type_commit(&segment_type);

  /* This is the initial communication stage which distributes the tiles to each process. Underneath this method uses a collective
   * scatter communication and hence all processes should call the method.
   */
  apsp_floyd_warshall_distribute_tiles(adjacency_matrix, n_vertices, tile_dim, tile_matrix_dim, 0, tile_world_comm, tile_type, &tile_i, &tile_j, tile);

#ifdef DEBUG
  if (my_rank == 0) {
    fprintf(stderr, "apsp_floyd_warshall: scattered the %i dimension displacement matrix into %i tiles of dimension %i\n", n_vertices, n_processes, tile_dim);
  }
  debug_print_matrix(stderr, "apsp_floyd_warshall: process %i is processing tile (%i, %i) =", tile, tile_dim, tile_dim, my_rank, tile_i, tile_j);
#endif // DEBUG

  int from_vertex_start, from_vertex_end, to_vertex_start, to_vertex_end;
  apsp_floyd_warshall_determine_tile_vertex_domain(n_vertices, tile_dim, tile_i, tile_j, &from_vertex_start, &from_vertex_end, &to_vertex_start, &to_vertex_end);

  int* kth_col_segment = calloc(tile_dim, sizeof(int));
  int* kth_row_segment = calloc(tile_dim, sizeof(int));

  for (int k = 0; k < n_vertices; k++) {
    apsp_floyd_warshall_recieve_required_kth_row_and_column_segments(k, my_rank, n_vertices, tile_matrix_dim, tile, tile_dim, tile_i, tile_j, tile_world_comm, tile_row_comms, tile_col_comms, segment_type, &kth_row_segment, &kth_col_segment);
#ifdef DEBUG
    debug_print_matrix(stderr, "apsp_floyd_warshall: process %i on iteration k = %i recieved col =", kth_col_segment, 1, tile_dim, my_rank, k);
    debug_print_matrix(stderr, "apsp_floyd_warshall: process %i on iteration k = %i recieved row =", kth_row_segment, 1, tile_dim, my_rank, k);
#endif // DEBUG
    /* from vertex is the 'row' */
    for (int from_vertex = from_vertex_start; from_vertex < from_vertex_end; from_vertex++) {
      /* to vertex is the 'col' */
      for (int to_vertex = to_vertex_start; to_vertex < to_vertex_end; to_vertex++) {
        const int d_ij_tile_index = ((from_vertex - from_vertex_start) * tile_dim) + (to_vertex - to_vertex_start);
        const int d_ij = tile[d_ij_tile_index];
        const int d_ik = kth_col_segment[from_vertex - from_vertex_start];
        const int d_kj = kth_row_segment[to_vertex - to_vertex_start];
        const int min_d_ij = min(d_ij, d_ik + d_kj);
#ifdef DEBUG
        fprintf(stderr, "apsp_floyd_warshall: process %i on iteration k = %i performing calc: (tile index=%i) d_%i,%i = min(d_%i,%i | d_%i,%i + d_%i,%i) = min(%i, %i + %i) = %i\n", my_rank, k, d_ij_tile_index, from_vertex, to_vertex, from_vertex, to_vertex, from_vertex, k, k, to_vertex, d_ij, d_ik, d_kj, min_d_ij);
#endif // DEBUG
        tile[d_ij_tile_index] = min_d_ij;
      }
    }
  }

  free(kth_col_segment);
  free(kth_row_segment);

#ifdef DEBUG
  fprintf(stderr, "apsp_floyd_warshall: process %i finished calculating tile", my_rank);
  debug_print_matrix(stderr, "apsp_floyd_warshall: process %i is calculated tile (%i, %i)", tile, tile_dim, tile_dim, my_rank, tile_i, tile_j);
#endif // DEBUG

  apsp_floyd_warshall_gather_tiles_to_root(adjacency_matrix, n_vertices, tile_dim, tile_matrix_dim, 0, tile, tile_world_comm);

  MPI_Type_free(&tile_type);
  MPI_Type_free(&segment_type);

  return 0;
}

void pause_to_allow_attachment()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    char answer[1];
    fprintf(stderr, "Press any key to continue...");
    fflush(stderr);
    fread_s(answer, sizeof(char) * _countof(answer), sizeof(char), _countof(answer), stdin);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

errno_t load_matrix_file(FILE* input_file, int* n_vertices, int** adjacency_matrix)
{
  if (input_file == NULL) {
    fprintf(stderr, "Input file cannot be null.");
    return EINVAL;
  }
  fread(n_vertices, sizeof(int), 1, input_file);
  const int n_adjacency_matrix_values = (*n_vertices) * (*n_vertices);
#ifdef DEBUG
  fprintf(stderr, "Allocating adjacency matrix with %i values (verts = %i)", n_adjacency_matrix_values, *n_vertices);
#endif // DEBUG
  * adjacency_matrix = calloc(n_adjacency_matrix_values, sizeof(int));
  if (*adjacency_matrix == NULL) {
    fprintf(stderr, "Failed to allocate memory for adjacency matrix.");
    return ENOMEM;
  }
  fread(*adjacency_matrix, sizeof(int), n_adjacency_matrix_values, input_file);
  return 0;
}

errno_t write_matrix_file(FILE* output_file, int as_text, int n_vertices, int* adjacency_matrix)
{
  if (output_file == NULL) {
    fprintf(stderr, "Output file cannot be null.");
    return EINVAL;
  }

  if (as_text) {
    fprintf(output_file, "%i", n_vertices);
    for (int i = 0; i < n_vertices * n_vertices; i++) {
      fprintf(output_file, " %i", adjacency_matrix[i]);
    }
    fprintf(output_file, "\n");
  }
  else {
    fwrite(&n_vertices, sizeof(int), 1, output_file);
    const int n_adjacency_values = n_vertices * n_vertices;
    fwrite(adjacency_matrix, sizeof(int), n_adjacency_values, output_file);
  }
  return 0;
}

void parse_cli_args(int argc, char* argv[], char** envp, char** maybe_input_filename, char** maybe_output_filename, int* output_as_text, int* include_timing_info)
{
  *maybe_input_filename = NULL;
  *maybe_output_filename = NULL;

  /* We start at 1 in order to skip the name of the program which is argv[0]. */
  for (int i = 1; i < argc; i++) {
    if (strcmp("-text", argv[i]) == 0) {
      *output_as_text = 1;
    }
    else if (strcmp("-time", argv[i]) == 0) {
      *include_timing_info = 1;
    }
    else if (*maybe_input_filename == NULL) {
      *maybe_input_filename = argv[i];
    }
    else if (*maybe_output_filename == NULL) {
      *maybe_output_filename = argv[i];
    }
  }
}

int main(int argc, char* argv[], char** envp)
{
  MPI_Init(&argc, &argv);

#ifdef DEBUG
  pause_to_allow_attachment();
#endif // DEBUG

  char* maybe_input_filename;
  char* maybe_output_filename;
  int output_as_text, include_timing_info;
  parse_cli_args(argc, argv, envp, &maybe_input_filename, &maybe_output_filename, &output_as_text, &include_timing_info);

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int n_vertices = -1;
  int* adjacency_matrix = NULL;

  if (my_rank == 0) {
    int close_input = 0;
    FILE* input_file = stdout;
    if (maybe_input_filename != NULL) {
      fopen_s(&input_file, maybe_input_filename, "rb");
      close_input = 1;
    }
    load_matrix_file(input_file, &n_vertices, &adjacency_matrix);
    if (close_input && input_file != NULL) {
      fclose(input_file);
    }
  }

  clock_t calc_start, calc_completed;
  calc_start = clock();
  apsp_floyd_warshall(adjacency_matrix, n_vertices);
  calc_completed = clock();

#ifdef DEBUG
  if (my_rank == 0) {
    debug_print_matrix(stderr, "apsp_floyd_warshall: final result =", adjacency_matrix, 1, 32);
}
#endif // DEBUG

  if (my_rank == 0) {
    if (adjacency_matrix == NULL) {
      fprintf(stderr, "No adjacency matrix to output.");
      return 1;
    }
    int close_output = 0;
    FILE* output_file = stdout;
    if (maybe_output_filename != NULL) {
      fopen_s(&output_file, maybe_output_filename, "w+");
      close_output = 1;
    }
    write_matrix_file(output_file, output_as_text, n_vertices, adjacency_matrix);
    if (close_output && output_file != NULL) {
      fclose(output_file);
    }
  }

  if (include_timing_info) {
    clock_t calc_duration = calc_completed - calc_start;
    fprintf(stdout, "time=%lf", ((double)calc_duration / (double)CLOCKS_PER_SEC) * 1000);
  }

  MPI_Finalize();

  return 0;
}