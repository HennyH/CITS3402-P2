# uwa-hpc-p2

# Building

Follow the instructions [here](https://blogs.technet.microsoft.com/windowshpc/2015/02/02/how-to-compile-and-run-a-simple-ms-mpi-program/) to install the MS-MPI SDK and executables. Ensure that after installation your `MSMPI_X` environment variables are set since the solution references these.

# Development

Install [CodeMaid](https://marketplace.visualstudio.com/items?itemName=SteveCadwallader.CodeMaid) and after it is installed enable formatting on save by clicking `Extensions > Code Main > Automatic Cleanup On Save ...`.

# Todo:

- Merge the get_kth_col/get_kth_row into one method and have it check set the buffer to all INT_NULLs if it doesn't exist (and then make sure it's freed and allocated once in the outer loop).
- Check all allocs/callocs are freed.
- Look for any TODOs.
- Do file I/O.
- Write python script to test program.

# References

1. [Parallelizing Floyd-Warshall](https://gkaracha.github.io/papers/floyd-warshall.pdf)
2. [Parallelizing Floyd-Warshall](https://en.wikipedia.org/wiki/Parallel_all-pairs_shortest_path_algorithm#Floyd_algorithm)
