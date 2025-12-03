from sklearn.cluster import KMeans, kmeans_plusplus
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .core import Imagefun


class Palette(Imagefun):
    image_palette_normalized: np.ndarray
    image_palette_colors: list

    def __init__(self):
        super().__init__()

    @staticmethod
    def find_closest_palette_color(pixel, palette: np.ndarray):
        """
        (static method)\n
        Find the closest color in the palette to a given pixel color.\n
        Uses Euclidean distance in RGB space.
        """
        distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
        return palette[np.argmin(distances)]

    @staticmethod
    def find_closest_palette_color_index(pixel, palette: np.ndarray):
        """
        (static method)\n
        Find the closest color in the palette to a given pixel color and return the index in the palette.\n
        Uses Euclidean distance in RGB space.
        """
        distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
        i = np.argmin(distances)
        return i, palette[i]

    @staticmethod
    def luminance(rgb):
        r, g, b = rgb
        if r > 1: # normalize
            r = r / 255
            g = g / 255
            b = b / 255
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def generate_optimized_palette(self, num_colors):
        """
        Generates an optimized color palette from an image using K-Means clustering.

        Args:
                num_colors (int): The number of colors for the new palette.

        Returns:
                np.ndarray: A numpy array of shape (num_colors, 3) representing the palette.
        """
        # Reshape the image to be a list of pixels
        img_array = np.array(self.image, dtype=np.float64) / 255
        pixels = img_array.reshape(-1, 3)

        # Use KMeans to find the most common colors
        # print(f"Generating a {num_colors}-color palette. This might take a moment...")
        # kmeans = KMeans(n_clusters=num_colors, algorithm='elkan', random_state=442).fit(pixels)
        # palette = kmeans.cluster_centers_

        palette, _ = kmeans_plusplus(pixels, num_colors, random_state=42)

        # print("Palette generated.")

        palette = [[j if j <= 1 else 1 for j in x] for x in palette]

        # convert to int (RGB)
        palette = [[j * 255 for j in x] for x in palette]

        # CORRECT MISPREDICTION
        # for colors in palette:
        #     print(colors)

        # # a simple approach: maybe 255 is not right...
        # palette = [[j if j < 254 else 0 for j in x] for x in palette]

        # another approach: scan the image pixels and check how many of the pixels are similar (within a boundary) to every color in the palette. if below some similarity drop the color.
        bound = 3
        colors_rejection = dict.fromkeys(range(len(palette)), 0)
        for y in tqdm(range(self.height)):
            for x in tqdm(range(self.width), leave=False):
                # compare the pixel with every color of the palette  # Sono un cane
                for i, color in enumerate(palette):
                    ok = []
                    for o, c in zip(self.pixels[x, y], color):
                        if o >= c - bound and o < c + bound:
                            ok.append(True)
                        else:
                            ok.append(False)

                    try:
                        ok.index(False)
                        colors_rejection[i] += 1
                    except ValueError:
                        pass
            #         break
            #     break
            # break
        total_pixels = self.height * self.width
        for c in colors_rejection:
            colors_rejection[c] = (colors_rejection[c] / total_pixels) * 100

        # print()
        # print(colors_rejection)

        # print()
        # for colors in palette:
        #     print(colors)

        # re-normalize palette
        palette = [[j / 255 for j in x] for x in palette]

        self.image_palette_normalized = palette
        self.image_palette_colors = [
            tuple(color) for color in np.asarray(palette).astype(int)
        ]

        return self

    def palette_2(self, num_colors):
        image = self.image.convert("RGB")
        img_array = np.array(image, dtype=float)
        # print(img_array.shape)
        # print(img_array[0][0])

        pixels = img_array.reshape(-1, 3)
        # print(pixels.shape)

        pixels = np.asarray([np.average(x) for x in pixels]).reshape(-1, 1)
        model = KMeans(n_clusters=num_colors, random_state=6759).fit(pixels)
        palette = model.cluster_centers_
        # print(palette)

        thresh = 3
        hits = dict.fromkeys(range(len(palette)), [])
        for y in tqdm(range(self.height)):
            for x in tqdm(range(self.width), leave=False):
                pixel = self.pixels[x, y]
                avg = np.average(pixel)
                for i, avg_color in enumerate(palette):
                    if np.abs(avg - avg_color) < thresh:
                        hits[i].append(pixel)
        # print([len(hits[x]) for x in hits])

        hits_clean = dict.fromkeys(range(len(palette)), [])
        for i in range(len(hits)):
            hits_clean[i] = list(set(hits[i]))[i]
        hits = hits_clean
        # print([len(hits[x]) for x in hits])

        self.image_palette_normalized = np.asarray(
            [np.asarray([j / 255 for j in hits[x]][0:3]) for x in hits]
        )
        return self

    def palette_3(self, num_colors):
        image = self.image.convert("RGB")
        img_array = np.array(image, dtype=float)

        pixels = img_array.reshape(-1, 3)

        if self.logger:
            self.logger.info(f"Starting KMeans palette generation with {num_colors} colors")
        pixels = np.asarray([np.average(x) for x in pixels]).reshape(-1, 1)
        model = KMeans(n_clusters=num_colors, random_state=6759).fit(pixels)
        palette = model.cluster_centers_
        if self.logger:
            self.logger.debug(
                f"Extracted Kmeans palette with {num_colors} colors",
                # extra={"palette": palette.tolist()},
            )

        has_white = False
        white_thres = 251
        thresh = 3
        hits = dict.fromkeys(range(len(palette)), [])
        for y in tqdm(range(self.height)):
            for x in tqdm(range(self.width), leave=False):
                pixel = self.pixels[x, y]
                avg = np.average(pixel)
                if avg > white_thres:
                    has_white = True
                for i, avg_color in enumerate(palette):
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

        if self.logger:
            self.logger.info(
                f"Final palette with {num_colors} colors",
                extra={"palette": palette.tolist()},
            )
        self.image_palette_normalized = palette / 255
        return self

    def plot_palette(self, output_name: str = None):
        """
        Plots the extracted color palette and saves to file.
        """

        # Normalize the RGB values to be between 0 and 1 for matplotlib
        # palette_normalized = np.array(self.image_palette_colors) / 255.0
        palette_normalized = self.image_palette_normalized

        # Create a figure and an axes object
        _, ax = plt.subplots(figsize=(len(palette_normalized), 1), dpi=80)

        # Display the colors using imshow. The input needs to be 3D.
        # We create an image of 1 pixel height and 'n_colors' width.
        ax.imshow([palette_normalized], aspect="auto")

        # Remove axes ticks and spines for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.title("Extracted Color Palette")

        if output_name != None:
            plt.savefig(output_name)
        else:
            plt.show()

        return self

    def set_palette(self, palette: list[list[float]]):
        # accepts both normalized and not-normalized paletted
        # if a color channel is > 1 it is not normalized (this is maybe a bit weak)
        is_norm = True
        for color in palette:
            for channel in color:
                if channel > 1:
                    is_norm = False
                    break

        palette = np.asarray(palette)

        if not is_norm:
            palette = palette / 255

        self.image_palette_normalized = palette
        return self
