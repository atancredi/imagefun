import numpy as np
from sklearn.cluster import MiniBatchKMeans
from logging import Logger
import numpy as np
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
    # 1. Keep native uint8 as long as possible to prevent massive memory spikes
    image = image.convert("RGB")
    img_array = np.array(image) 
    pixels = img_array.reshape(-1, 3)

    # 2. Faster sampling using the modern NumPy Generator API
    if len(pixels) > sample_pixels:
        rng = np.random.default_rng(6759)
        # Sampling directly on the axis is faster than generating indices
        sample = rng.choice(pixels, size=sample_pixels, replace=False, axis=0)
    else:
        sample = pixels

    # 3. Convert ONLY the sample to float32 (cuts memory overhead in half vs float64)
    sample_float = sample.astype(np.float32)

    if logger:
        logger.info(f"Starting palette generation with {num_colors} colors")

    # 4. Use MiniBatchKMeans: The ultimate speedup for color quantization
    kmeans = MiniBatchKMeans(
        n_clusters=num_colors,
        random_state=6759,
        n_init="auto",
        batch_size=2048, # Fast batch sizes
        max_iter=15      # Cut down max iterations; colors converge quickly
    ).fit(sample_float)
    
    palette = np.clip(np.round(kmeans.cluster_centers_), 0, 255).astype(np.float32)

    if logger:
        logger.debug(f"Extracted palette with {num_colors} colors")

    # 5. DETECT WHITE & BLACK PRESENCE IN SAMPLE ONLY (Massive compute save)
    brightness = sample_float.mean(axis=1)
    has_white = np.any(brightness > 240)
    has_black = np.any(brightness < 20)

    # 6. CLEANUP: Vectorized, using squared distances to avoid slow square roots
    MIN_DIST_SQ = 18 * 18 
    cleaned = [palette[0]]
    
    for c in palette[1:]:
        # Compare current color against all already-cleaned colors simultaneously
        dists_sq = np.sum((np.array(cleaned) - c) ** 2, axis=1)
        if np.all(dists_sq > MIN_DIST_SQ):
            cleaned.append(c)
            
    palette = np.array(cleaned)

    # If too few colors remain, refill from original cluster centers
    if len(palette) < num_colors:
        for c in kmeans.cluster_centers_:
            if len(palette) >= num_colors:
                break
            dists_sq = np.sum((palette - c) ** 2, axis=1)
            if np.all(dists_sq > 64): # 8 squared
                palette = np.vstack([palette, c])

    # 7. WHITE / BLACK CONSISTENCY (Vectorized check)
    if has_white:
        if not np.any(np.sum((palette - np.array([255., 255., 255.])) ** 2, axis=1) < 625): # 25 squared
            palette = np.vstack([palette, [255., 255., 255.]])
    if has_black:
        if not np.any(np.sum((palette - np.array([0., 0., 0.])) ** 2, axis=1) < 625):
            palette = np.vstack([palette, [0., 0., 0.]])

    # FINAL SIZE ADJUSTMENT
    if len(palette) > num_colors:
        palette = palette[np.argsort(palette.mean(axis=1))][:num_colors]
        
    if len(palette) < num_colors:
        needed = num_colors - len(palette)
        gray = np.linspace(40, 220, needed).reshape(-1, 1)
        filler = np.repeat(gray, 3, axis=1)
        palette = np.vstack([palette, filler])

    # Convert to integer
    palette = np.clip(np.round(palette), 0, 255).astype(int)

    if with_percentages:
        # Assuming get_palette_percentages exists in your scope
        palette_percentages = get_palette_percentages(image, palette)
        palette = [np.concatenate((x, [palette_percentages[i]])) for i, x in enumerate(palette)]

    if logger:
        logger.info(
            f"Extracted final palette with {num_colors} colors",
            extra={"palette": palette.tolist()},
        )
        
    return palette
