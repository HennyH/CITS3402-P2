
import sys
import os
import unittest
import subprocess
import io
import csv
import argparse
import random
from traceback import print_exc
from math import sqrt, ceil
from tempfile import TemporaryFile
from itertools import product, chain

from test_program import MPI_EXEC, PROGRAM, int_bytes, INF

params_multi = (
    [1, 2, 4, 8, 16],
    [256, 364, 512, 768, 1024, 1436, 2048, 4096, 5120]
)

def measure(csv_writer, test_name, n_threads, infile, sample_number):
    try:
        result = subprocess.run(
            [MPI_EXEC, "-n", str(n_threads), PROGRAM, infile, "-time"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
    except subprocess.CalledProcessError:
        print(f"Error measuring ({n_threads}, {infile}):", file=sys.stderr)
        print_exc(file=sys.stderr)
        return None

    maybe_timing_info = result.stdout.splitlines()[-1]
    if "time=" not in maybe_timing_info:
        print(f"No timing info found in stderr: {result.stderr}", file=sys.stderr)
        return None

    timing_info = float(maybe_timing_info.lstrip("time="))
    size = int(os.path.basename(infile).rstrip(".in"))
    csv_writer.writerow([test_name, sample_number, n_threads, size, timing_info])
    print(f"Gathered measurement ({sample_number}, {n_threads}, {size}) = {maybe_timing_info}", file=sys.stderr)
    return None

def make_test_matrix(size):
    filename = f"./inputs/{size}.in"
    if not os.path.exists(filename):
        with open(filename, "wb+") as matrix:
            matrix.write(int_bytes(size))
            for i in range(size):
                for j in range(size):
                    if random.choice([True, False]):
                        matrix.write(int_bytes(random.randint(0, 1000)))
                    else:
                        matrix.write(int_bytes(0))
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Performance Measurement Tool")
    parser.add_argument("--output",
                        type=argparse.FileType("w+"),
                        default=sys.stdout)
    parser.add_argument("--samples",
                        type=int,
                        default=50)
    result = parser.parse_args(sys.argv[1:])
    csv_writer = csv.writer(result.output)
    csv_writer.writerow(["test", "sample", "theads", "size", "time"])

    #tests = \
    #    chain((("multi", max(5, int(result.samples / 2)), *x) for x in product(*params_multi)),
    #          (("speedup_4t", result.samples, *x) for x in product(*params_multi_4_proc_speedup)),
    #          (("speedup_2048d", result.samples, *x) for x in product(*params_2048_speedup)))
    tests = (("multi", max(5, int(result.samples / 2)), *x) for x in product(*params_multi))
    for test_name, samples, n_threads, size in tests:
        infile = make_test_matrix(size)
        for sample_n in range(1, samples + 1):
            measure(csv_writer, test_name, n_threads, infile, sample_n)
            result.output.flush()
