from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from numba import njit

from ..protocol import ImagefunProtocol

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core import Imagefun


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
    dists = np.sum(diff**2, axis=2)

    # index of closest color for each pixel
    closest_indices = np.argmin(dists, axis=1)

    # occurrences of each palette index
    counts = np.bincount(closest_indices, minlength=len(palette))

    # percentages
    total_pixels = pixel_array.shape[0]
    percentages = counts / total_pixels

    return percentages


# numba optimized
@njit(fastmath=True)
def _atkinson_dither_core(img_array, palette_norm):
    height, width, _ = img_array.shape
    num_colors = palette_norm.shape[0]

    for r in range(height):
        for c in range(width):
            # read pixels
            p0 = img_array[r, c, 0]
            p1 = img_array[r, c, 1]
            p2 = img_array[r, c, 2]

            # closest palette color with Inlined Euclidean distance
            min_dist = 1e8
            best_idx = 0
            for i in range(num_colors):
                d0 = p0 - palette_norm[i, 0]
                d1 = p1 - palette_norm[i, 1]
                d2 = p2 - palette_norm[i, 2]

                # Squared distance
                dist = d0 * d0 + d1 * d1 + d2 * d2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

            new_p0 = palette_norm[best_idx, 0]
            new_p1 = palette_norm[best_idx, 1]
            new_p2 = palette_norm[best_idx, 2]

            img_array[r, c, 0] = new_p0
            img_array[r, c, 1] = new_p1
            img_array[r, c, 2] = new_p2

            # quantization error
            err0 = (p0 - new_p0) * 0.125
            err1 = (p1 - new_p1) * 0.125
            err2 = (p2 - new_p2) * 0.125

            # error diffusion
            if c + 1 < width:
                v0 = img_array[r, c + 1, 0] + err0
                img_array[r, c + 1, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                v1 = img_array[r, c + 1, 1] + err1
                img_array[r, c + 1, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                v2 = img_array[r, c + 1, 2] + err2
                img_array[r, c + 1, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)

            if c + 2 < width:
                v0 = img_array[r, c + 2, 0] + err0
                img_array[r, c + 2, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                v1 = img_array[r, c + 2, 1] + err1
                img_array[r, c + 2, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                v2 = img_array[r, c + 2, 2] + err2
                img_array[r, c + 2, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)

            if r + 1 < height:
                if c - 1 >= 0:
                    v0 = img_array[r + 1, c - 1, 0] + err0
                    img_array[r + 1, c - 1, 0] = (
                        0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                    )
                    v1 = img_array[r + 1, c - 1, 1] + err1
                    img_array[r + 1, c - 1, 1] = (
                        0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                    )
                    v2 = img_array[r + 1, c - 1, 2] + err2
                    img_array[r + 1, c - 1, 2] = (
                        0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)
                    )

                v0 = img_array[r + 1, c, 0] + err0
                img_array[r + 1, c, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                v1 = img_array[r + 1, c, 1] + err1
                img_array[r + 1, c, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                v2 = img_array[r + 1, c, 2] + err2
                img_array[r + 1, c, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)

                if c + 1 < width:
                    v0 = img_array[r + 1, c + 1, 0] + err0
                    img_array[r + 1, c + 1, 0] = (
                        0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                    )
                    v1 = img_array[r + 1, c + 1, 1] + err1
                    img_array[r + 1, c + 1, 1] = (
                        0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                    )
                    v2 = img_array[r + 1, c + 1, 2] + err2
                    img_array[r + 1, c + 1, 2] = (
                        0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)
                    )

            if r + 2 < height:
                v0 = img_array[r + 2, c, 0] + err0
                img_array[r + 2, c, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                v1 = img_array[r + 2, c, 1] + err1
                img_array[r + 2, c, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                v2 = img_array[r + 2, c, 2] + err2
                img_array[r + 2, c, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)

    return img_array


class ImagefunDitheringMixin(ImagefunProtocol):

    palette: np.ndarray
    palette_dominance: np.ndarray

    def get_palette(self, num_colors=8, sample_pixels=500000) -> "Imagefun":
        self.log(f"Starting palette generation with {num_colors} colors")

        image_rgb = self.image.convert("RGB")
        pixels = np.array(image_rgb).reshape(-1, 3)

        if len(pixels) > sample_pixels:
            rng = np.random.default_rng(6759)
            sample = rng.choice(pixels, size=sample_pixels, replace=False, axis=0)
        else:
            sample = pixels

        sample_float = sample.astype(np.float32)

        # 2x clusters for more choice in selecting colors for better contrast
        n_clusters = min(num_colors * 2, len(sample_float))

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=6759,
            n_init="auto",
            batch_size=2048,
            max_iter=15,
        ).fit(sample_float)

        candidates = np.clip(np.round(kmeans.cluster_centers_), 0, 255)

        # sort by dominance
        counts = np.bincount(kmeans.labels_, minlength=n_clusters)
        candidates = candidates[np.argsort(counts)[::-1]]

        # detect black and white
        brightness = sample_float.mean(axis=1)
        has_white = np.any(brightness > 240)
        has_black = np.any(brightness < 20)

        selected_colors = []

        # If black/white exist, anchor them first. This naturally forces the algorithm
        # to pick subsequent colors that are far away from pure black/white!
        if has_black:
            selected_colors.append(np.array([0.0, 0.0, 0.0]))
        if has_white:
            selected_colors.append(np.array([255.0, 255.0, 255.0]))

        # If no black/white, anchor with the most dominant color in the image
        if not selected_colors:
            selected_colors.append(candidates[0])
            candidates = np.delete(candidates, 0, axis=0)

        # Iteratively pick the candidate furthest from already selected colors
        while len(selected_colors) < num_colors and len(candidates) > 0:
            selected_arr = np.array(selected_colors)

            # Fast vectorized squared distance between all candidates and all selected colors
            dists_sq = np.sum(
                (candidates[:, np.newaxis, :] - selected_arr[np.newaxis, :, :]) ** 2,
                axis=2,
            )

            # Find the distance to the CLOSEST selected color for each candidate
            min_dists_sq = np.min(dists_sq, axis=1)

            # Pick the candidate where this minimum distance is the MAXIMUM (Farthest Point)
            best_idx = np.argmax(min_dists_sq)

            selected_colors.append(candidates[best_idx])
            candidates = np.delete(candidates, best_idx, axis=0)

        palette = np.array(selected_colors)

        # fallback
        if len(palette) < num_colors:
            needed = num_colors - len(palette)
            filler = np.repeat(np.linspace(40, 220, needed).reshape(-1, 1), 3, axis=1)
            palette = np.vstack([palette, filler])

        palette = np.clip(np.round(palette[:num_colors]), 0, 255).astype(int)
        palette_dominance = get_palette_percentages(image_rgb, palette)

        self.log(
            f"Extracted final palette with {num_colors} colors",
            extra={"palette": palette.tolist()},
        )

        self.palette = palette
        self.palette_dominance = palette_dominance

        return self

    @property
    def palette_colors(self):
        return self.palette.tolist()

    @property
    def palette_normalized(self):
        return self.palette / 255

    def dithering(
        self,
    ):
        palette = self.palette_normalized

        # check for numba
        palette_norm = np.ascontiguousarray(palette, dtype=np.float64)

        image = self.image.convert("RGB")
        img_array = np.array(image, dtype=np.float64) / 255.0

        print("Applying Atkinson error diffusion...")
        img_array = _atkinson_dither_core(img_array, palette_norm)

        self.image = Image.fromarray((img_array * 255).astype(np.uint8), "RGB")

        return self
