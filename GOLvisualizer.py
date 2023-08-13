import pyaudiowpatch as pyaudio
import sys
import pygame
import time
from math import sqrt
import numpy as np


def neighbours(current_x, current_y, area):
    """
    Counts the number of surrounding alive neighbours
    # [-1][-1], [0][-1], [1][-1],
    # [-1][0],  [0][0],  [1][0],
    # [-1][1],  [0][1],  [1][1]

    :param current_x:
    :param current_y:
    :param area:
    :return: neighbour_count
    """
    neighbour_count = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            if x == 0 and y == 0:
                continue
            elif area[current_x + x][current_y + y]:
                neighbour_count += 1
    return neighbour_count


def is_alive(x, y, area):
    """
    Decides whether the cell is alive based on standard game of life rules
    :param x:
    :param y:
    :param area:
    :return:
    """
    neighbour_count = neighbours(x, y, area)

    if area[x][y]:
        if 1 < neighbour_count < 4:
            return True
    else:
        if neighbour_count == 3:
            return True
    return False


MULTIPLIER = 2  # Defines the size of cells on screen
RATE = 44100  # Sampling rate dependent on device
CHUNK = int((1 / 60) * RATE)  # 1/30
FORMAT = pyaudio.paInt16
SCREEN_HEIGHT = 150
SCREEN_WIDTH = int(CHUNK / 2)  # The chunk contains 4 phases, half of one suffices


def main():
    # pyaudio setup
    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            p.get_device_info_by_host_api_device_index(0, i).get('name')
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)

    # Get default WASAPI speakers
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if not default_speakers["isLoopbackDevice"]:
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                default_speakers = loopback
                break
        else:
            print("Get default WASAPI speakers")
            exit()

    stream = p.open(format=pyaudio.paInt16,
                    channels=default_speakers["maxInputChannels"],
                    rate=int(default_speakers["defaultSampleRate"]),
                    frames_per_buffer=CHUNK,
                    input=True,
                    input_device_index=default_speakers["index"]
                    )

    # cell arrays setup
    #  + 2 allows for border buffer zone for neighbours()
    cells_current = [[False for i in range(int(SCREEN_HEIGHT) + 2)] for j in range(int(SCREEN_WIDTH) + 2)]
    cells_next = [[False for i in range(int(SCREEN_HEIGHT) + 2)] for j in range(int(SCREEN_WIDTH) + 2)]

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH * MULTIPLIER, SCREEN_HEIGHT * MULTIPLIER))
    clock = pygame.time.Clock()
    running = True
    time.sleep(2)
    screen.fill("black")

    size_x = len(cells_current)
    size_y = len(cells_current[0])

    pygame.display.flip()
    time.sleep(1)
    # main process/draw loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # process next frame
        for x in range(0, size_x - 1):
            for y in range(0, size_y - 1):
                cells_next[x][y] = is_alive(x, y, cells_current)

        buff = stream.read(CHUNK)
        data = np.frombuffer(buff, dtype=np.int16)
        fft_complex = np.fft.fft(data, n=CHUNK)  # fast furrier transform
        max_val = sqrt(max(v.real * v.real + v.imag * v.imag for v in fft_complex)) + 1
        # max of real numbers in max_val, +1 to avoid division by 0
        scale_value = SCREEN_HEIGHT / max_val

        # draw cells
        screen.fill("black")

        for x in range(0, size_x - 1):
            for y in range(0, size_y - 1):
                if cells_next[x][y]:
                    pygame.draw.rect(screen, "white", (x * MULTIPLIER, y * MULTIPLIER, MULTIPLIER, MULTIPLIER))
        # draw frequencies
        s = 0
        for i, v in enumerate(fft_complex):
            dist = sqrt(v.real * v.real + v.imag * v.imag)
            mapped_dist = dist * scale_value + 2
            s += mapped_dist

            if i < size_x:
                cells_next[i][int(SCREEN_HEIGHT - mapped_dist)] = True

            pygame.draw.rect(screen,
                             "green",
                             ((i, SCREEN_HEIGHT * MULTIPLIER - mapped_dist), (MULTIPLIER, int(SCREEN_HEIGHT * 3/2)))
                             )
        # flip cells
        for x in range(0, size_x - 1):
            for y in range(0, size_y - 1):
                cells_current[x][y] = cells_next[x][y]

        pygame.display.flip()

        clock.tick(30)


if __name__ == '__main__':
    sys.exit(main())
