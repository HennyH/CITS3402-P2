import csv
import statistics
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt

TEST = 0
SAMPLE = 1
THREAD = 2
SIZE = 3
TIME = 4

with open("measurements_limited - Copy.csv", "r") as csv_fileobj:
    reader = csv.reader(csv_fileobj)
    next(reader)
    measurements = sorted([(n, int(i), int(t), int(s), float(d)) for (n, i, t, s, d) in reader],
                          key=itemgetter(SIZE))
    n_threads = len(list(set(map(itemgetter(THREAD), measurements))))
    print("n threads", n_threads)
    for size, size_entries in groupby(measurements, key=itemgetter(SIZE)):
        size_entries = sorted(list(size_entries), key=itemgetter(THREAD))
        prev_mean = None
        prev_stdev = None
        for i, (thread, thread_entries) in enumerate(groupby(size_entries, key=itemgetter(THREAD)), start=1):
            thread_entries = list(thread_entries)
            times = list(map(itemgetter(TIME), thread_entries))
            mean = statistics.mean(times)
            stdev = statistics.stdev(times)
            #time_sets.append(times)
            #time_set_labels.append(f"{size}x{size} Matrix")
            mean_diff_pc = None if prev_mean is None else (mean / prev_mean - 1)
            mean_stdev_pc = None if prev_stdev is None else (stdev / prev_stdev - 1)
            print(f"threads={thread},size={size} => mean={mean}, {mean_diff_pc}%, stdev={stdev}, {mean_stdev_pc}%")
            prev_mean = mean
            prev_stdev = stdev
        #ax = plt.subplot(n_threads, 1, i)
        #ax.boxplot(time_sets, showfliers=False, vert=False, labels=time_set_labels)
        #ax.set_title(f"{thread} Thread Performance Scaling")
