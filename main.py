# Author: Teshan Liyanage <teshanuka@gmail.com>

import pygame
import numpy as np
from time import time

from shape import BuildShape, to_shape_arr, BuildShapeMO
from bot import Bot
from env import Env
from utils import print_array

colors = pygame.color.THECOLORS

IMG_PATH = "data/star.png"
IMG_SHAPE = (40, 30)
SC_SIZE = (800, 600)
FPS = 120


def print_map(arr):
    print_array(arr, val_map={np.inf: '\u2588', 1: ' '})


def test_bot_stationary():
    """bot appears"""
    pygame.init()
    clock = pygame.time.Clock()

    env = Env(SC_SIZE, ((120, 120), (750, 550)), IMG_SHAPE)
    bot = Bot((50, 50), 1, env, BuildShape(to_shape_arr(IMG_PATH, IMG_SHAPE)))
    done = False

    while not done:
        for _event in pygame.event.get():
            if _event.type == pygame.QUIT:
                done = True

        env.blit()
        bot.draw()
        pygame.display.flip()
        clock.tick(FPS)


def test_bot_move():
    """bot goes to the direction of closest pickup point (non stop)"""
    pygame.init()
    clock = pygame.time.Clock()

    env = Env(SC_SIZE, ((120, 120), (750, 550)), IMG_SHAPE)
    bot = Bot((50, 50), 1, env, BuildShape(to_shape_arr(IMG_PATH, IMG_SHAPE)))
    done = False

    bot.set_pickup_path()

    while not done:
        for _event in pygame.event.get():
            if _event.type == pygame.QUIT:
                done = True

        env.blit()
        bot.advance()
        bot.draw()
        pygame.display.flip()
        clock.tick(FPS)


def test_bot_one_block_shape(pos: tuple = (4, 4)):
    """bot goes to the direction of closest pickup point (non stop)"""
    img_arr = np.zeros((8, 6)).T
    img_arr[pos] = 1

    print_array(img_arr)

    pygame.init()
    clock = pygame.time.Clock()

    env = Env(SC_SIZE, ((120, 120), (750, 550)), (8, 6))
    bot = Bot((50, 50), 1, env, BuildShape(to_shape_arr(img_arr)))

    done = False

    while not done:
        for _event in pygame.event.get():
            if _event.type == pygame.QUIT:
                done = True

        env.blit()
        bot.step()
        bot.draw()
        if len(bot.path) > 1:
            pygame.draw.lines(env.screen, colors['yellow'], False, [env.bpos2mpos(p) for p in bot.path])

        pygame.display.flip()
        clock.tick(FPS)


def test_bot_task():
    """bot goes to the direction of closest pickup point (non stop)"""
    pygame.init()
    clock = pygame.time.Clock()

    env = Env(SC_SIZE, ((120, 120), (750, 550)), IMG_SHAPE)
    bot = Bot((50, 50), 1, env, BuildShape(to_shape_arr(IMG_PATH, IMG_SHAPE), get_method="middle_out"))
    done = False

    while not done:
        for _event in pygame.event.get():
            if _event.type == pygame.QUIT:
                done = True

        env.blit()
        bot.step()
        bot.draw()

        if len(bot.path) > 1:
            pygame.draw.lines(env.screen, colors['yellow'], False, [env.bpos2mpos(p) for p in bot.path])

        pygame.display.flip()
        clock.tick(FPS)


def test_bot_task_middle_out():
    """bot goes to the direction of closest pickup point (non stop)"""
    pygame.init()
    clock = pygame.time.Clock()

    env = Env(SC_SIZE, ((120, 120), (750, 550)), IMG_SHAPE)
    bot = Bot((50, 50), 1, env, BuildShapeMO(to_shape_arr(IMG_PATH, IMG_SHAPE)))
    done = False

    while not done:
        for _event in pygame.event.get():
            if _event.type == pygame.QUIT:
                done = True

        env.blit()
        bot.step()
        bot.draw()

        if len(bot.path) > 1:
            pygame.draw.lines(env.screen, colors['yellow'], False, [env.bpos2mpos(p) for p in bot.path])

        pygame.display.flip()
        clock.tick(FPS)

    print_array(bot._build_shape.orig_array)
    print_map(bot._build_shape.global_map)


def test_bots_task(n):
    """bot goes to the direction of closest pickup point (non stop)"""
    pygame.init()
    clock = pygame.time.Clock()

    fo = 50
    factories = ((fo, fo), (SC_SIZE[0]-fo, fo), (fo, SC_SIZE[1]-fo), (SC_SIZE[0]-fo, SC_SIZE[1]-fo))
    env = Env(SC_SIZE, factories, IMG_SHAPE)
    bots = [Bot(np.random.randint((0, 0), SC_SIZE), 1, env,
                BuildShape(to_shape_arr(IMG_PATH, IMG_SHAPE), get_method='reachable')) for _ in range(n)]
    done = False
    paused = False

    while not done:
        for _event in pygame.event.get():
            if _event.type == pygame.QUIT:
                done = True
            if _event.type == pygame.KEYDOWN:
                if _event.key == pygame.K_SPACE:
                    paused = ~paused

        if paused:
            clock.tick(FPS)
            continue

        env.blit()
        if all(bot._state == "dead" or bot._build_shape.finished for bot in bots):
            print("FINISHED")
            paused = True
            continue

        for bot in bots:
            bot.step()
            bot.draw()

            try:
                if len(bot.path) > 1:
                    pygame.draw.lines(env.screen, bot._color, False, [env.bpos2mpos(p) for p in bot.path])
                if bot._state == "drop":
                    pygame.draw.circle(env.screen, bot._color, env.bpos2mpos(bot._drop_pos), 5)
            except TypeError:
                pass

        pygame.display.flip()
        clock.tick(FPS)

    print_array(bots[0]._build_shape.orig_array)
    print_map(bots[0]._build_shape.global_map)


def test_bots_task_middle_out(n):
    """bot goes to the direction of closest pickup point (non stop)"""
    pygame.init()
    clock = pygame.time.Clock()

    fo = 50
    factories = ((fo, fo), (SC_SIZE[0]-fo, fo), (fo, SC_SIZE[1]-fo), (SC_SIZE[0]-fo, SC_SIZE[1]-fo))
    env = Env(SC_SIZE, factories, IMG_SHAPE)
    bots = [Bot(np.random.randint((0, 0), SC_SIZE), 1, env,
                BuildShapeMO(to_shape_arr(IMG_PATH, IMG_SHAPE))) for _ in range(n)]
    done = False
    paused = False

    while not done:
        for _event in pygame.event.get():
            if _event.type == pygame.QUIT:
                done = True
            if _event.type == pygame.KEYDOWN:
                if _event.key == pygame.K_SPACE:
                    paused = ~paused

        if paused:
            clock.tick(FPS)
            continue

        env.blit()
        if all(bot.done for bot in bots):
            print("FINISHED")
            paused = True
            continue

        for bot in bots:
            bot.step()
            bot.draw()

            try:
                if len(bot.path) > 1:
                    pygame.draw.lines(env.screen, bot._color, False, [env.bpos2mpos(p) for p in bot.path])
                if bot._state == "drop":
                    pygame.draw.circle(env.screen, bot._color, env.bpos2mpos(bot._drop_pos), 5)
            except TypeError:
                pass

        pygame.display.flip()
        clock.tick(FPS)

    print_map(bots[0]._build_shape.global_map)
    
    
def run_sim(env, n):
    """bot goes to the direction of closest pickup point (non stop)"""
    pygame.init()
    clock = pygame.time.Clock()

    bots = [Bot(np.random.randint((0, 0), SC_SIZE), 1, env,
                BuildShapeMO(to_shape_arr(IMG_PATH, IMG_SHAPE))) for _ in range(n)]
    done = False

    t0 = time()
    loop_cnt = 0
    while not done:
        #for _event in pygame.event.get():
        #    if _event.type == pygame.QUIT:
        #        done = True

        env.blit()
        if all(bot.done for bot in bots):
            print("FINISHED")
            done = True
            
        for bot in bots:
            bot.step()
            bot.draw()

            try:
                if len(bot.path) > 1:
                    pygame.draw.lines(env.screen, bot._color, False, [env.bpos2mpos(p) for p in bot.path])
                if bot._state == "drop":
                    pygame.draw.circle(env.screen, bot._color, env.bpos2mpos(bot._drop_pos), 5)
            except TypeError:
                pass

        pygame.display.flip()
        clock.tick(FPS)
        loop_cnt += 1

    et = time() - t0
    print(f"[TIME] {n} bots {et:.2f}s {loop_cnt}")

    return loop_cnt, et
    
    
def num_bots_vs_perf(min_bots=81, max_bots=100, outfile="num_bots_vs_perf.csv"):
    import matplotlib.pyplot as plt
    
    fo = 50
    factories = ((fo, fo), (SC_SIZE[0]-fo, fo), (fo, SC_SIZE[1]-fo), (SC_SIZE[0]-fo, SC_SIZE[1]-fo))
    env = Env(SC_SIZE, factories, IMG_SHAPE)
    
    times = []
    with open(outfile, 'w') as fp:
        for n_bots in range(min_bots, max_bots+1):
            loop_cnt, et = run_sim(env, n_bots)
            
            fp.write(f"{n_bots},{et},{loop_cnt}\n")
            times.append(et)
            
            env.reset()
            
    plt.plot(range(min_bots, max_bots+1), times)
    plt.show()


if __name__ == '__main__':
    # test_bot_stationary()
    # test_bot_move()
    # test_bot_one_block_shape()
    # test_bot_task()
    # test_bot_task_middle_out()
    # test_bots_task(5)
    test_bots_task_middle_out(5)
    
    # num_bots_vs_perf()
