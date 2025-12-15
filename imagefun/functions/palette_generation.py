from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from logging import Logger

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


def generate_palette_old(image: Image.Image, num_colors: int, logger: Logger = None):
    image = image.convert("RGB")
    # img_array = np.array(image, dtype=float)
    # pixels = img_array.reshape(-1, 3)

    # pixels = image.load()
    pixels = np.asarray(image)
    height = image.height
    width = image.width

    if logger:
        logger.info(f"Starting KMeans palette generation with {num_colors} colors")
    pixels = np.asarray([np.average(x) for x in pixels]).reshape(-1, 1)
    model = KMeans(n_clusters=num_colors, random_state=6759).fit(pixels)
    palette = model.cluster_centers_
    if logger:
        logger.debug(
            f"Extracted Kmeans palette with {num_colors} colors",
            # extra={"palette": palette.tolist()},
        )

    has_white = False
    white_thres = 251
    thresh = 3
    hits = dict.fromkeys(range(len(palette)), [])
    pixels = image.load()
    for y in tqdm(range(height)):
        for x in tqdm(range(width), leave=False):
            pixel = pixels[x, y]
            avg = np.average(pixel)
            if avg > white_thres:
                has_white = True
            for i, avg_color in enumerate([np.average(x) for x in palette]):
                if np.absolute(avg - avg_color) < thresh:
                    hits[i].append(pixel)

    hits_clean = dict.fromkeys(range(len(palette)), [])
    for i in range(len(hits)):
        hits_clean[i] = list(set(hits[i]))[i]
    hits = hits_clean

    palette = np.asarray([np.asarray(hits[x][0:3]) for x in hits])
    avg_palette = [np.average(color) for color in palette]

    palette_has_white = False
    try:
        p = [a > white_thres for a in avg_palette]
        p.index(True)
        palette_has_white = True
    except ValueError:
        pass

    # are there two similar colors?
    similar_colors = []
    sim_thresh = 1
    for i, c1 in enumerate(avg_palette):
        for j, c2 in enumerate(avg_palette):
            if np.absolute(c1 - c2) < sim_thresh:
                similar_colors.append((i, j))
    similar_colors = list(
        set([tuple(sorted(x)) for x in similar_colors if x[0] != x[1]])
    )  # NOSONAR
    # if similar colors and the palette has white (undetected) substitute
    if len(similar_colors) > 0 and has_white and not palette_has_white:
        palette[similar_colors[0][0]] = [255.0, 255.0, 255.0]
        # palette[similar_colors[0][0]] = [0.,0.,0.]
        # XXX HERE THERE IS A PROBLEAM
        # white is rarely recognized as closest color to pixel, so a special thresh for white detection must be applied

    # print(similar_colors)

    if logger:
        logger.info(
            f"Final palette with {num_colors} colors",
            extra={"palette": palette.tolist()},
        )
    return palette


def generate_palette(image: Image.Image, num_colors=8, sample_pixels=500000, logger: Logger = None):

    image = image.convert("RGB")
    img_array = np.array(image, dtype=float)
    pixels = img_array.reshape(-1, 3)

    # Random pixel sample for speed
    if len(pixels) > sample_pixels:
        idx = np.random.choice(len(pixels), sample_pixels, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    if logger:
        logger.info(f"Starting palette generation with {num_colors} colors")
    kmeans = KMeans(
        n_clusters=num_colors,
        random_state=6759,
        n_init="auto"
    ).fit(sample)
    palette = kmeans.cluster_centers_
    palette = np.clip(np.round(palette), 0, 255).astype(float)
    if logger:
        logger.debug(
            f"Extracted palette with {num_colors} colors",
        )

    # DETECT WHITE & BLACK PRESENCE IN IMAGE
    brightness = pixels.mean(axis=1)
    has_white = np.any(brightness > 240)
    has_black = np.any(brightness < 20)
    # CLEANUP: MERGE NEAR-DUPLICATE COLORS
    MIN_DIST = 18
    cleaned = []
    for c in palette:
        if all(np.linalg.norm(c - cc) > MIN_DIST for cc in cleaned):
            cleaned.append(c)
    palette = np.array(cleaned)
    # If too few colors remain, refill from original cluster centers
    if len(palette) < num_colors:
        full = kmeans.cluster_centers_
        for c in full:
            if len(palette) >= num_colors:
                break
            if all(np.linalg.norm(c - p) > 8 for p in palette):
                palette = np.vstack([palette, c])
    # WHITE / BLACK CONSISTENCY
    if has_white:
        if not np.any(np.linalg.norm(palette - np.array([255, 255, 255])) < 25):
            palette = np.vstack([palette, [255, 255, 255]])
    if has_black:
        if not np.any(np.linalg.norm(palette - np.array([0, 0, 0])) < 25):
            palette = np.vstack([palette, [0, 0, 0]])
    # FINAL SIZE ADJUSTMENT
    # Too many colors → sort by brightness and trim
    if len(palette) > num_colors:
        palette = palette[np.argsort(palette.mean(axis=1))][:num_colors]
    # Too few colors → pad with grayscale
    if len(palette) < num_colors:
        needed = num_colors - len(palette)
        gray = np.linspace(40, 220, needed).reshape(-1, 1)
        filler = np.repeat(gray, 3, axis=1)
        palette = np.vstack([palette, filler])

    # convert to integer
    palette = np.clip(np.round(palette), 0, 255).astype(int)

    # XXX do i really need to sort by mean?
    # i've included a sorting on luminance in the embedding of the palette
    palette = palette[np.argsort(palette.mean(axis=1))]

    if logger:
        logger.info(
            f"Extracted final palette with {num_colors} colors",
            extra={"palette": palette.tolist()},
        )
    return palette

