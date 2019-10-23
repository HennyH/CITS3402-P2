# uwa-hpc-p2

# Contents

1. `HPCP2.Program/Program.c` - The parallelized implementation of the Floyd-Warshall algorithim.
2. `HPCP2.Testing/test_program.py` - A Python script which when run test the compiled C program (you must build the solution in **Release mode** before you run this!).
3. `HPCP2.Testing/floyd_warshall.py` - A simple reference Python implementation of the floyd-warshall algorithim to asset correctness.
4. `HPCP2.Testing/measure_performance.py` - Python script which gathers timing information of various thread count and sizes of the compiled program.
5. `HPCP2.Testing/graph_measurements.py` - Calculates summary stats from the measurements gathered from the `measurement_performance.py`.
6. `Report.pdf` - The PDF of the report.
7. `executable_win10_x64.zip` - A zip file with the compiled program for 64-bit Windows 10.

# Building

Follow the instructions [here](https://blogs.technet.microsoft.com/windowshpc/2015/02/02/how-to-compile-and-run-a-simple-ms-mpi-program/) to install the MS-MPI SDK and executables. Ensure that after installation your `MSMPI_X` environment variables are set since the solution references these.

The project should be already be configured such that:
1. Configuration > C/C++ has "$(MSMPI_INC);$(MSMPI_INC)\x64" added to the 'Additional Include Directories'.
2. Configuration > Linker has "$(MSMPI_LIB64)" added to the 'Additional Library Directories'.

However if not, these settings can be changed by right clicking 'HPCP2.Program' from the Solution Explorer and selecting 'Properties'.

# Development

Install [CodeMaid](https://marketplace.visualstudio.com/items?itemName=SteveCadwallader.CodeMaid) and after it is installed enable formatting on save by clicking `Extensions > Code Main > Automatic Cleanup On Save ...`.
