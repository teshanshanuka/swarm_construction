import matplotlib.pyplot as plt
import numpy as np


def rolling_median(x, n):
    x = np.array(x)
    idx = np.arange(n) + np.arange(len(x) - n + 1)[:, None]
    b = [row[row>0] for row in x[idx]]
    return np.array(list(map(np.median, b)))


def readfile(fname):
    xs, ts, lcs = [], [], []
    with open(fname, 'r') as fp:
        for line in fp:
            x, y, lc = line.strip().split(' ')
            xs.append(int(x)); ts.append(float(y[:-1])); lcs.append(int(lc))
    return xs, ts, lcs


def plot(xs, ts, lcs):
    ts = rolling_median(ts, 5)
    plt.plot(ts, label='time')
    # plt.plot(xs, [_l/max(lcs) for _l in lcs], label='loop count')
    plt.xlabel('Num bots')
    plt.ylabel('Time steps')
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    xs, ys, lcs = readfile("num_bots_vs_performance.txt")
    plot(xs, ys, lcs)
