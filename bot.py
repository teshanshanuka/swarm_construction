# Author: Teshan Liyanage <teshanuka@gmail.com>

import numpy as np
import pygame
from pyastar import reduce_path, astar_path as _astar_path

from shape import BuildShape
from env import Env
from utils import neighborhood, parallel_vec, line_pixels

colors = pygame.color.THECOLORS

pygame.font.init()
bot_font = pygame.font.SysFont('Comic Sans MS', 30)

S_IDLE = "idle"
S_PICKUP = "pickup"
S_DROP = "drop"
S_HOMING = "homing"
S_DEAD = "dead"


def astar_path(arr, start, end):
    path_t = _astar_path(arr, start, end, allow_diagonal=True)
    if path_t is not None:
        return reduce_path(path_t)


class Bot:
    _id = 0

    def __init__(self, pos, vel: float, env: Env, build_shape: BuildShape):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array([0, 0], dtype=np.float64)

        self._vel_mag = vel
        self._build_shape = build_shape
        self._color = np.random.randint(0, 256, 3)
        self._env = env

        self._id = Bot._id
        Bot._id += 1

        self._state = S_IDLE
        self._goal_idx = -1
        self._goal_reached = False
        self._has_block = False
        self._drop_pos = None
        self.path = None
        self._next_goal_idx = -1

        self.bpos = self._bpos

    def __repr__(self):
        return f"Bot {self._id}"

    @property
    def _bpos(self) -> tuple:
        return self._env.mpos2bpos(self.pos)

    @property
    def done(self) -> bool:
        return self._state == S_DEAD or self._build_shape.finished

    def load(self, unload_pos: tuple) -> bool:
        if not self.set_unload_path(unload_pos):
            print(f"No unload path for {self} to {unload_pos}")
            self._build_shape.local_map[unload_pos] = np.inf
            return False
        self._has_block = True
        return True

    def unload(self) -> tuple:
        self._has_block = False
        self._env.place_block(self._drop_pos)
        print(f"{self} unloading on {self._drop_pos}")

        self._build_shape.fill(self._drop_pos)
        return self._drop_pos

    def set_pickup_path(self):
        load_bot_pos = self._env.get_closest_block_factory(self.bpos)
        return self.set_path(load_bot_pos)

    def set_unload_path(self, unload_pos: tuple) -> bool:
        assert unload_pos is not None
        _unload_bot_pos = self._build_shape.get_closest_block(unload_pos, self.bpos)
        if _unload_bot_pos is None:
            return False

        if not self.set_path(_unload_bot_pos):
            print(f"[WARN] {self} cannot find a path")
            return False

        self._drop_pos = unload_pos
        return True

    def set_path(self, pos: tuple):
        # print(f"Setting path to {pos} current {self.bpos}")
        if pos == self.bpos:
            self._goal_reached = True
            return True
        self.path = astar_path(self._build_shape.local_map, self.bpos, pos)
        if self.path is None:
            print(f"{self} - no valid A* path to {self.bpos}-{self._env.global_map[self.bpos]} {pos}-{self._env.global_map[pos]}")
            self._build_shape.fill(pos, fill_global=False)
            return False

        self._goal_idx = len(self.path) - 1
        self._goal_reached = False
        self._next_goal_idx = 1
        self.set_vel(self.path[self._next_goal_idx])
        return True

    def update_maps(self):
        r = 2
        nbhood = neighborhood(self._env.global_map, self.bpos, r)
        start_pos = (max(0, self.bpos[0] - r), max(0, self.bpos[1] - r))
        for a_pos, val in np.ndenumerate(nbhood):
            fill_pos = (a_pos[0] + start_pos[0], a_pos[1] + start_pos[1])
            try:
                if val == np.inf and not self._build_shape.occupied(fill_pos):
                    self._build_shape.fill(fill_pos)
            except KeyError:
                pass

    def is_stuck(self) -> bool:
        nbrs_mask = ((-1, 0), (0, 1), (1, 0), (0, -1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1))
        for nbr_loc in nbrs_mask:
            nbr = (self.bpos[0] + nbr_loc[0], self.bpos[1] + nbr_loc[1])
            # print(f"Checking nbr {nbr}")
            sx, sy = self._env.global_map.shape
            if (0 <= nbr[0] < sx) and (0 <= nbr[1] < sy):  # If neighbor is within array bounds
                if self._env.global_map[nbr] != np.inf:  # If neighbor is not occupied
                    return False

        print(f"[WARN] {self} is stuck at {self.bpos}")
        return True

    def set_vel(self, to_pos):
        to_mpos = self._env.bpos2mpos(to_pos)
        self.vel = parallel_vec((to_mpos[0] - self.pos[0], to_mpos[1] - self.pos[1]), self._vel_mag)

    def advance(self):
        self.pos += self.vel

        if self.reached(self.path[self._next_goal_idx]):
            if self._next_goal_idx == self._goal_idx:
                # print(f"Goal {self.path[self._next_goal_idx]}")
                self._goal_reached = True
                return
            # print(f"got to {self._path[self._next_goal_idx]} next {self._path[self._next_goal_idx+1]}")
            self._next_goal_idx += 1
            self.set_vel(self.path[self._next_goal_idx])

        path_poss = line_pixels(self.bpos, self.path[self._next_goal_idx])
        for pos in path_poss:
            if self._env.is_occupied(pos):
                if not self.set_path(tuple(self.path[-1])):
                    print(f"[WARN] {self} Stuck while re routing")
                    self._state = S_IDLE
                break

    def step(self):
        if self._state == S_DEAD:
            return

        if self._build_shape.finished:
            return

        if self._state == S_HOMING and self._goal_reached:
            self._state = S_IDLE
            print(f"{self} idling")
            return

        self.bpos = self._bpos

        if self._state != S_HOMING and self._build_shape.finished:
            print(f"{self} finished")
            if not self.set_pickup_path():
                print(f"[WARN] {self} Can't find way back")
                self._state = S_DEAD
                return
            self._state = S_HOMING
            return

        self.update_maps()

        if self.is_stuck():
            self._state = S_DEAD
            return

        if self._state == S_PICKUP and self._goal_reached:
            next_block = self._build_shape.get_next_block(self.bpos)
            if self._build_shape.finished:
                print(f"{self} finished while pickup")
                return
            if not self.load(next_block):
                self._state = S_IDLE
                return
            self._state = S_DROP
            return

        if self._state == S_DROP and self._goal_reached:
            if self._env.is_occupied(self._drop_pos):
                print(f"{self} abort drop")
                self._state = S_IDLE
                return
            self.unload()
            self._state = S_IDLE
            return

        if self._state == S_IDLE and not self._has_block and not self._build_shape.finished:
            self.set_pickup_path()
            self._state = S_PICKUP
            return

        if self._state == S_IDLE and self._has_block and not self._build_shape.finished:
            next_block = self._build_shape.get_next_block(self.bpos)
            if self._build_shape.finished:
                print(f"{self} finished while rerouting")
                return
            if not self.load(next_block):
                self._state = S_IDLE
                return
            self._state = S_DROP
            return

        self.advance()

    def reached(self, pos):
        block_mpos = self._env.bpos2mpos(pos)
        return np.linalg.norm((self.pos[0] - block_mpos[0], self.pos[1] - block_mpos[1])) < 1

    def draw(self, r=10):
        pygame.draw.circle(self._env.screen, self._color if self._state != S_DEAD else colors['brown'],
                           self.pos.astype(int), r)
        if self._has_block:
            pygame.draw.rect(self._env.screen, colors['black'], (*(self.pos - r / 2).astype(int), r, r))

        # if self._state == S_DEAD:
        ts = bot_font.render(str(self._id), False, colors['black'])
        self._env.screen.blit(ts, self.pos.astype(int))
