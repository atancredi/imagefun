from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

from .core import Imagefun


class Palette(Imagefun):
    image_palette_normalized: np.ndarray
    image_palette_colors: list


    def __init__(self, properties=None):
        super().__init__(properties)


    @staticmethod
    def find_closest_palette_color(pixel, palette):
        """
        (static method)\n
        Find the closest color in the palette to a given pixel color.\n
        Uses Euclidean distance in RGB space.
        """
        distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
        return palette[np.argmin(distances)]


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
        print(f"Generating a {num_colors}-color palette. This might take a moment...")
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10).fit(pixels)

        # The cluster centers are our new palette
        palette = kmeans.cluster_centers_
        print("Palette generated.")
        self.image_palette_normalized = palette
        self.image_palette_colors = [tuple(color) for color in palette.astype(int)]
        return self


    def plot_palette(self, output_name: str):
        """
        Plots the extracted color palette and saves to file.
        """
        if not self.image_palette_colors:
            print("Cannot plot an empty palette.")
            return

        # Normalize the RGB values to be between 0 and 1 for matplotlib
        palette_normalized = np.array(self.image_palette_colors) / 255.0

        # Create a figure and an axes object
        _, ax = plt.subplots(figsize=(len(self.image_palette_colors), 1), dpi=80)

        # Display the colors using imshow. The input needs to be 3D.
        # We create an image of 1 pixel height and 'n_colors' width.
        ax.imshow([palette_normalized], aspect="auto")

        # Remove axes ticks and spines for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.title("Extracted Color Palette")
        plt.savefig(output_name)
