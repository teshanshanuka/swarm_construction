# Author: Teshan Liyanage <teshanuka@gmail.com>

import numpy as np

from typing import Union


def rotate(vec, angle):
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    return np.dot(rot_matrix, vec)


def to_vec(mag: float, angle: float):
    """Create a vector of size `mag` at `angle`"""
    return mag * np.array([np.cos(angle), np.sin(angle)], dtype='float64')


def parallel_vec(vec: Union[tuple, np.ndarray], mag: float):
    """Get vector of size `mag` in the direction of vec"""
    return to_vec(mag, np.arctan2(vec[1], vec[0]))


def neighborhood(arr: np.ndarray, pos: tuple, n: int):
    """Slice neighborhood of `pos` from `arr`"""
    sx, sy = max(0, pos[0] - n), max(0, pos[1] - n)
    ex, ey = min(pos[0] + n + 1, arr.shape[0]), min(pos[1] + n + 1, arr.shape[1])
    return arr[sx: ex, sy: ey]


def merge(big_arr: np.ndarray, smol_arr: np.ndarray, pos: tuple, offset: int = 0):
    """Put `smol_arr` in `big_arr` at position `pos` with an offset `offset`"""
    pos = (max(0, pos[0] - offset), max(0, pos[1] - offset))
    for _a_pos, val in np.ndenumerate(smol_arr):
        big_arr[_a_pos[0] + pos[0], _a_pos[1] + pos[1]] = val


def print_array(arr, val_map={1: '\u2588', 0: ' '}, w=2, print_func=print):
    print_func("┌" + "─" * arr.shape[1] * w + "┐")
    for row in arr:
        s = "│" + "".join(val_map[v] * w for v in row) + "│"
        print_func(s)
    print_func("└" + "─" * arr.shape[1] * w + "┘")


def print_arrays(arrays, names=None, val_map={1: '\u2588', 0: ' '}, w=2, sep=" ", print_func=print):
    assert len(set(arr.shape for arr in arrays)) == 1, "arrays in the array list must have similar shapes"

    if names:
        print_func(sep.join(f"{name:^{arrays[0].shape[1] * w + 2}}" for name in names))
    print_func(sep.join("┌" + "─" * arr.shape[1] * w + "┐" for arr in arrays))
    for i in range(arrays[0].shape[0]):
        print_func(sep.join("│" + "".join(val_map[v] * w for v in arr[i]) + "│" for arr in arrays))
    print_func(sep.join("└" + "─" * arr.shape[1] * w + "┘" for arr in arrays))


def reduce_path2(path: np.ndarray):
    diff = np.diff(path, axis=0)
    mask = np.where((diff != (0, 0)).any(axis=1))[0]

    _path = np.concatenate([path[mask], [path[-1]]])
    ddiff = np.diff(np.diff(_path, axis=0), axis=0)
    mask = np.where((ddiff != (0, 0)).any(axis=1))[0] + 1

    return np.concatenate([[_path[0]], _path[mask], [_path[-1]]])


def line_pixels(start: tuple, end: tuple):
    # https://stackoverflow.com/questions/4381269/line-rasterisation-cover-all-pixels-regardless-of-line-gradient
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy  # e2 is error value e_xy

    ret = []
    while True:
        ret.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        # horizontal step?
        if e2 > dy:
            err += dy
            x0 += sx

        # vertical step?
        if e2 < dx:
            err += dx
            y0 += sy

    return ret


def centroid(arr: np.ndarray) -> tuple:
    _x = (np.arange(arr.shape[1]) * arr.sum(axis=0)).sum() / arr.sum()
    _y = (np.arange(arr.shape[0]) * arr.sum(axis=1)).sum() / arr.sum()
    return int(_y), int(_x)


def map_array(arr: np.ndarray, map_dict: dict):
    return np.vectorize(map_dict.get)(arr)
