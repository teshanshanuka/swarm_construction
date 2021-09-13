# Author: Teshan Liyanage <teshanl@zone24x7.com>

import pygame
import numpy as np

from shape import BaseShape
colors = pygame.color.THECOLORS


class Env:
    _intd = False

    def __init__(self, size: tuple, block_factories: tuple, img_size: tuple):
        assert not Env._intd, "Singleton class"

        self.size = size
        self.env_arr = np.ones(size, np.float32)

        self.global_map = BaseShape(shape=img_size)
        self.block_size = (self.size[0] // self.global_map.shape[0],
                           self.size[1] // self.global_map.shape[1])
        print(f"Env size={self.size} map size={self.global_map.shape} block size={self.block_size}")

        self.screen = pygame.display.set_mode(size)
        self.block_factories = tuple(self.mpos2bpos(bf) for bf in block_factories)

        self._placed_blocks = []
        self._block_factory_rects = tuple(self.get_block_rect(p) for p in self.block_factories)

        Env._intd = True
        
    def reset(self):
        self.global_map.reset()
        self._placed_blocks = []

    def bpos2mpos(self, pos) -> tuple:
        """Convert block position to map position"""
        return (pos[0] * self.block_size[0] + self.block_size[0] // 2,
                pos[1] * self.block_size[1] + self.block_size[1] // 2)

    def mpos2bpos(self, pos) -> tuple:
        """Convert map position to block position"""
        return int(pos[0] // self.block_size[0]), int(pos[1] // self.block_size[1])

    def get_closest_block_factory(self, pos: tuple):
        dists = [np.linalg.norm((x - pos[0], y - pos[1])) for x, y in self.block_factories]
        return self.block_factories[np.argmin(dists)]

    def is_occupied(self, pos: tuple):
        return self.global_map[pos] == np.inf

    def get_block_rect(self, pos: tuple):
        sx, sy = pos[0] * self.block_size[0], pos[1] * self.block_size[1]
        return sx, sy, *self.block_size

    def place_block(self, pos: tuple):
        self._placed_blocks.append(self.get_block_rect(pos))
        self.global_map[pos] = np.inf

    def blit(self):
        self.screen.fill(colors['grey68'])
        for b in self._placed_blocks:
            pygame.draw.rect(self.screen, colors['black'], b)
        for b in self._block_factory_rects:
            pygame.draw.rect(self.screen, colors['red'], b)
            pygame.draw.rect(self.screen, colors['brown'], b, 10)
