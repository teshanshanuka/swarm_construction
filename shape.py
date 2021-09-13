# Author: Teshan Liyanage <teshanuka@gmail.com>

import numpy as np

from utils import print_array, print_arrays, centroid, map_array


class BaseShape:
    global_map: np.array
    shape: tuple

    def __init__(self, arr=None, shape=None):
        if arr is not None:
            try:
                assert arr.shape == BaseShape.shape, "All shapes should be of the same size"
            except AttributeError:
                BaseShape.global_map = np.ones_like(arr, dtype=np.float32)
                BaseShape.shape = self.global_map.shape
        elif shape is not None:
            BaseShape.global_map = np.ones(shape, dtype=np.float32)
            BaseShape.shape = shape
        else:
            raise RuntimeError("Need either `arr` or `shape` to initialize `BaseShape`")

    @classmethod
    def global_fill(cls, pos: tuple):
        cls.global_map[pos] = np.inf

    @classmethod
    def global_is_occupied(cls, pos: tuple):
        return cls.global_map[pos] == np.inf
        
    def reset(self):
        BaseShape.global_map = np.ones(BaseShape.shape, dtype=np.float32)

    def __getitem__(self, item):
        return self.global_map.__getitem__(item)

    def __setitem__(self, key, value):
        self.global_map.__setitem__(key, value)


class BuildShape(BaseShape):
    def __init__(self, arr: np.ndarray = None, get_method='natural'):
        """
        :param get_method: `get_next_block()` function returns the next block to be placed depending on this method
            natural - iterate over the shape in the array order
            closest - next closest block to the bot position
            reachable - next reachable block (not obstructed by surrounding blocks)
        :param arr: Array marking the shape with elements of 1/True
        """
        assert len(arr.shape) == 2, "Expected a 2D array"

        self._get_method = get_method
        self.array = arr.T
        self.orig_array = self.array.copy()
        super().__init__(self.array)

        self.finished = False

        self.local_map = np.ones_like(self.array, dtype=np.float32)
        self._nbrs_mask = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    @property
    def nbrs_mask(self):
        np.random.shuffle(self._nbrs_mask)
        return self._nbrs_mask

    def __getitem__(self, item):
        return self.local_map.__getitem__(item)

    def fill(self, pos: tuple, fill_global=True):
        self.array[pos] = 0
        self.local_map[pos] = np.inf
        if fill_global:
            self.global_fill(pos)

    def clear(self, pos: tuple):
        self.array[pos] = 1
        self.local_map[pos] = 1

    def occupied(self, pos: tuple) -> bool:
        return self.local_map[pos] == np.inf

    def _surrounded(self, pos) -> bool:
        for nbr_loc in self.nbrs_mask:
            nbr = (pos[0] + nbr_loc[0], pos[1] + nbr_loc[1])
            if (0 <= nbr[0] < self.shape[0]) and (0 <= nbr[1] < self.shape[1]):  # If neighbor is within array bounds
                if not self.occupied(nbr):  # If neighbor is not occupied, pos is reachable
                    return False
        return True

    def get_next_block(self, *args):
        pos = None
        if self._get_method == 'natural':
            for _pos, v in np.ndenumerate(self.array):
                if v:
                    pos = _pos
        elif self._get_method == 'closest':
            assert len(args) == 1 and len(args[0]) == 2, "Bot position must be passed as an argument"
            bot_pos = args[0]
            dists = []
            for _pos, v in np.ndenumerate(self.array):
                if v:
                    dists.append((np.linalg.norm((_pos[0] - bot_pos[0], _pos[1] - bot_pos[1])), _pos))

            if dists:
                _, pos = min(dists, key=lambda a: a[0])
        elif self._get_method == 'reachable':
            assert len(args) == 1 and len(args[0]) == 2, "Bot position must be passed as an argument"
            bot_pos = args[0]
            dists = []
            for _pos, v in np.ndenumerate(self.array):
                if v and not self._surrounded(_pos):
                    dists.append((np.linalg.norm((_pos[0] - bot_pos[0], _pos[1] - bot_pos[1])), _pos))
            if dists:
                _, pos = min(dists, key=lambda a: a[0])
        else:
            raise RuntimeError(f"Get method '{self._get_method}' is not recognized")
        if pos is not None:
            return pos
        self.finished = True

    def get_closest_block(self, block_pos, bot_pos):
        """Get closest empty block position to the block_pos from bot_pos"""
        nbrs = []
        for idx, nbr_loc in enumerate(self.nbrs_mask):
            nbr = (block_pos[0] + nbr_loc[0], block_pos[1] + nbr_loc[1])
            if (0 <= nbr[0] < self.shape[0]) and (0 <= nbr[1] < self.shape[1]) and self.local_map[nbr] == 1:
                nbrs.append((nbr, np.linalg.norm((bot_pos[0] - nbr[0], bot_pos[1] - nbr[1]))))
        if nbrs:
            nbrs.sort(key=lambda x: x[1])
            return nbrs[0][0]


class BuildShapeMO(BuildShape):
    """Build from middle out"""

    def __init__(self, arr: np.ndarray = None):
        """
        :param arr: Array marking the shape with elements of 1/True
        """
        super().__init__(arr)
        self._next_block_candidates = None

    def _nbrs(self, pos, outside: bool = True, condition=lambda x: True) -> list:
        nbrs = []
        for nbr_loc in self.nbrs_mask:
            nbr = (pos[0] + nbr_loc[0], pos[1] + nbr_loc[1])
            if not (0 <= nbr[0] < self.shape[0]) or not (0 <= nbr[1] < self.shape[1]):
                if outside:
                    nbrs.append(nbr)
            elif condition(nbr):  # If neighbor is occupied, or out of range, count it in
                nbrs.append(nbr)

        return nbrs

    def _occupied_nbrs(self, pos, outside: bool = True) -> list:
        return self._nbrs(pos, outside, lambda x: self.occupied(x))

    def _free_nbrs(self, pos, outside: bool = True) -> list:
        return self._nbrs(pos, outside, lambda x: not self.occupied(x))

    def fill(self, pos: tuple, fill_global=True):
        self.array[pos] = 0
        self.local_map[pos] = np.inf
        if fill_global:
            self.global_fill(pos)
        self._next_block_candidates = self._free_nbrs(pos, outside=False)

    def get_next_block(self, *args):
        pos = None

        assert len(args) == 1 and len(args[0]) == 2, "Bot position must be passed as an argument"
        bot_pos = args[0]

        ct = centroid(self.orig_array)
        if not self.occupied(ct):  # Put block on centroid
            pos = ct

        else:
            candidates = []
            nb_candidates = []

            for _pos, v in np.ndenumerate(self.array):
                if self.occupied(_pos):
                    continue

                if v and (nbr_cnt := len(self._occupied_nbrs(_pos))) > 0:
                    if self._next_block_candidates and _pos in self._next_block_candidates:
                        nb_candidates.append((_pos, nbr_cnt))
                    candidates.append((_pos, nbr_cnt))  # Position and how many occupied neighboring blocks

            if nb_candidates:  # If there are neighborhood candidates, no need to get all
                candidates = nb_candidates

            if candidates:
                _, max_nbrs = max(candidates, key=lambda c: c[1])  # Max nbrs a candidate have

                local_ct = centroid(map_array(self.local_map, {np.inf: 1, 1: 0}))  # Centroid of local map
                dists = [(np.linalg.norm((_p[0] - local_ct[0], _p[1] - local_ct[1])), _p) for _p, _c in candidates if
                         _c == max_nbrs]

                if dists:
                    _, pos = min(dists, key=lambda d: d[0])

        if pos is not None:
            return pos

        self.finished = True

    def get_closest_block(self, block_pos, bot_pos):
        """Get closest empty block position to the block_pos from bot_pos"""
        nbrs = []
        for idx, nbr_loc in enumerate(self.nbrs_mask):
            nbr = (block_pos[0] + nbr_loc[0], block_pos[1] + nbr_loc[1])
            if (0 <= nbr[0] < self.shape[0]) and (0 <= nbr[1] < self.shape[1]) and self.local_map[nbr] == 1:
                nbrs.append((nbr, np.linalg.norm((bot_pos[0] - nbr[0], bot_pos[1] - nbr[1]))))
        if nbrs:
            nbrs.sort(key=lambda x: x[1])
            return nbrs[0][0]


def to_shape_arr(source, resize=None):
    if isinstance(source, np.ndarray):
        return source.copy()

    if isinstance(source, str):
        import os
        assert os.path.isfile(source), f"Did not find a source image at '{source}'"

        from PIL import Image
        img = Image.open(source).convert('L')
        if resize is not None:
            img = img.resize(resize)
        array = np.copy(np.asarray(img))
        array[array <= 255 // 2] = 1
        array[array > 255 // 2] = 0
        return array

    raise RuntimeError(f"Source '{source}' is not understood")


map_print_map = {np.inf: '\u2588', 1: ' '}

if __name__ == '__main__':
    arr_ = np.zeros((4, 5), dtype=int)
    arr_[1, 1] = 1
    arr_[1, 2] = 1
    arr_[2, 2] = 1

    b1 = BuildShape(arr_.copy())
    b2 = BuildShape(arr_.copy())

    print('build map')
    print_array(b1.array)

    b1.fill((1, 1))
    b2.fill((2, 2))

    print_arrays([b1.local_map, b2.local_map], ['b1', 'b2'], val_map=map_print_map)
    print('global map')
    print_array(b1.global_map, val_map=map_print_map)

    assert b1.get_closest_block((1, 2), (1, 0)) == (0, 2)
    while (nb := b1.get_next_block((0, 1))) is not None:
        b1.fill(nb)

    print("b1 filled")
    print_array(b1.local_map, val_map=map_print_map)

    arr_ = to_shape_arr("data/star.png", resize=(20, 25))
    print("Star")
    print_array(arr_)
