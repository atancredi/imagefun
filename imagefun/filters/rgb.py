import numpy as np


# rgb filter WITH PARAM
def saturation(pixel, amount: int):
    r, g, b = pixel
    newpixel = [0, 0, 0]
    oldpixel = (r, g, b)
    average = sum(oldpixel) // 3
    for i in range(3):
        newpixel[i] = int(max(min((oldpixel[i] - average) * amount + average, 255), 0))
    return newpixel[0], newpixel[1], newpixel[2]


# rgb filter
def invert_channels(pixel):
    r, g, b = pixel
    return b, r, g


# rgb filter
def rebalance_levels(pixel):
    r, g, b = pixel
    return int(min(r * 1.9, 255)), g, int(b * 0.1)


# rgb filter
def make_darker(pixel):
    r, g, b = pixel
    # make the dark darker
    avg = np.average((r, g, b))
    if avg < 100:
        r = r - 50
        g = g - 50
        b = b - 50
    return r, g, b


# rgb filter
def make_darker_2(pixel):
    r, g, b = pixel
    # make the dark darker
    avg = np.average((r, g, b))
    n = 5
    m = 4
    p = 2
    if avg < 100:
        r = r - (n * 10)
        g = g - (n * 10)
        b = b - (n * 10)
    elif avg < 150:
        r = r - (m * 10)
        g = g - (m * 10)
        b = b - (m * 10)
    elif avg < 200:
        r = r - (p * 10)
        g = g - (p * 10)
        b = b - (p * 10)
    return r, g, b


# rgb filter
def cross_bwand(pixel):
    r, g, b = pixel
    # fico
    g &= r
    b &= r
    return r, g, b


# rgb filter WITH PARAM
def bwand(pixel, bit):
    r, g, b = pixel
    r &= bit
    g &= bit
    b &= bit
    return r, g, b
