from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from logging import Logger

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


def get_palette_percentages(image: Image.Image, palette: list, resize_dim=(150, 150)):

    # resize image, with nearest neighbour resampling for keeping the structure
    img_small = image.resize(resize_dim, resample=Image.Resampling.NEAREST)
    
    img_array = np.array(img_small.convert("RGB"))

    pixel_array = img_array.reshape(-1, 3)
    palette_array = np.array(palette)

    # euclidean distance from every pixel to palette color
    # numpy broadcast hack
    diff = pixel_array[:, np.newaxis, :] - palette_array[np.newaxis, :, :]
    
    # squared euclidean distance
    dists = np.sum(diff ** 2, axis=2)

    # index of closest color for each pixel
    closest_indices = np.argmin(dists, axis=1)

    # occurrences of each palette index
    counts = np.bincount(closest_indices, minlength=len(palette))

    # percentages
    total_pixels = pixel_array.shape[0]
    percentages = counts / total_pixels

    return percentages.tolist()


def generate_palette(image: Image.Image, num_colors=8, sample_pixels=500000, logger: Logger = None, with_percentages=False):

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

    if with_percentages:
        # now get the percentage of image occupied by each color
        palette_percentages = get_palette_percentages(image, palette)
        palette = [np.concat((x, [palette_percentages[i]])) for i,x in enumerate(palette)]

    if logger:
        logger.info(
            f"Extracted final palette with {num_colors} colors",
            extra={"palette": palette.tolist()},
        )
    return palette

