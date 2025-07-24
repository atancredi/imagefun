import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .core import Imagefun
from .palette import Palette


class Dithering(Imagefun, Palette):

    def __init__(self, properties = None):
        super(Imagefun).__init__(properties)


    @staticmethod
    def get_bayer_matrix(size):
        """
        Generates a Bayer matrix of a given size (must be a power of 2).
        
        Args:
            size (int): The size of the matrix (e.g., 2, 4, 8).

        Returns:
            np.ndarray: The generated Bayer matrix.
        """
        if size == 2:
            return np.array([[0, 2], [3, 1]])
        else:
            sub_matrix = Dithering.get_bayer_matrix(size // 2)
            return np.block([
                [4 * sub_matrix,     4 * sub_matrix + 2],
                [4 * sub_matrix + 3, 4 * sub_matrix + 1]
            ])


    def get_palette(self, palette, bw):
        # --- Define Palettes ---
        if bw:
            # Black and White palette
            palette = np.array([[0, 0, 0], [1, 1, 1]])
        elif palette is None:
            # Default 8-color RGB palette
            palette = np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            ])
        else:
            # User-defined palette
            palette = np.array(palette, dtype=float) / 255.0
        return palette


    # Dithering algorithms
    def threshold(self, palette=None, bw=False):
        # Simple thresholding
        palette = self.get_palette(palette, bw)
        img_array = self.image.convert("RGB")
        
        if bw:
            grayscale = np.mean(img_array, axis=2)
            threshold_val = 0.5
            dithered_array = (grayscale > threshold_val).astype(float)
            # Convert back to 3 channels for consistent output format
            img_array = np.stack([dithered_array]*3, axis=-1)
        else: # RGB
             for r in range(self.height):
                for c in range(self.width):
                    img_array[r, c] = self.find_closest_palette_color(img_array[r, c], self.image_palette_normalized)
        
        if len(img_array) > 0:
            self.image = Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')
        else:
            print("WARNING: img array in threshold dithering has size 0")
        
        return self


    def random(self, bw=False):
        # Random dithering
        img_array = self.image.convert("RGB")
        random_noise = np.random.random((self.height, self.width))
        
        if bw:
            grayscale = np.mean(img_array, axis=2)
            dithered_array = (grayscale > random_noise).astype(float)
            img_array = np.stack([dithered_array]*3, axis=-1)
        else: # RGB
            # Add noise to each channel independently
            random_noise_rgb = np.random.random(img_array.shape)
            noisy_img = np.clip(img_array + (random_noise_rgb - 0.5) * 0.5, 0, 1)
            for r in range(self.height):
                for c in range(self.width):
                    img_array[r, c] = self.find_closest_palette_color(noisy_img[r, c], self.image_palette_normalized)
        
        if len(img_array) > 0:
            self.image = Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')
        else:
            print("WARNING: img array in random dithering has size 0")
        
        return self


    def bayer(self, bayer_size=4, bw=False):
        # Ordered dithering with a Bayer matrix
        if bayer_size not in [2, 4, 8]:
            raise ValueError("Bayer matrix size must be 2, 4, or 8.")
        
        bayer_matrix = self.get_bayer_matrix(bayer_size)
        bayer_norm = bayer_matrix / (bayer_size**2) - 0.5
        
        img_array = self.image.convert("RGB")
        
        tq1 = tqdm(range(self.height))
        tq1.set_description_str("bayer dithering")
        tq2 = tqdm(range(self.width), leave=False)
        for r in tq1:
            for c in tq2:
                threshold = bayer_norm[r % bayer_size, c % bayer_size]
                if bw:
                    intensity = np.mean(img_array[r, c])
                    img_array[r, c] = 1.0 if intensity + threshold > 0.5 else 0.0
                else: # RGB
                    new_pixel = np.clip(img_array[r, c] + threshold, 0, 1)
                    img_array[r, c] = self.find_closest_palette_color(new_pixel, self.image_palette_normalized)

        if len(img_array) > 0:
            self.image = Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')
        else:
            print("WARNING: img array in bayer matrix dithering has size 0")
        
        return self


    def floyd_steinberg(self):
        # Error-diffusion dithering
        img_array = self.image.convert("RGB")

        error_dist = [((0, 1), 7/16), ((1, -1), 3/16), ((1, 0), 5/16), ((1, 1), 1/16)]

        tq1 = tqdm(range(self.height))
        tq1.set_description_str("floyd steinberg dithering")
        tq2 = tqdm(range(self.width), leave=False)
        for r in tq1:
            for c in tq2:
                old_pixel = img_array[r, c].copy()
                new_pixel = self.find_closest_palette_color(old_pixel, self.image_palette_normalized)
                img_array[r, c] = new_pixel
                
                quant_error = old_pixel - new_pixel

                for (dr, dc), factor in error_dist:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        img_array[nr, nc] = np.clip(img_array[nr, nc] + quant_error * factor, 0, 1)

        if len(img_array) > 0:
            self.image = Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')
        else:
            print("WARNING: img array in floyd-steinberg dithering has size 0")
        
        return self


    def diffusion(self):
        # Error-diffusion dithering
        img_array = self.image.convert("RGB")

        error_dist = [((0, 1), 1/8), ((0, 2), 1/8), ((1, -1), 1/8), 
                          ((1, 0), 1/8), ((1, 1), 1/8), ((2, 0), 1/8)] # Atkinson

        tq1 = tqdm(range(self.height))
        tq1.set_description_str("error diffusion dithering")
        tq2 = tqdm(range(self.width), leave=False)
        for r in tq1:
            for c in tq2:
                old_pixel = img_array[r, c].copy()
                new_pixel = self.find_closest_palette_color(old_pixel, self.image_palette_normalized)
                img_array[r, c] = new_pixel
                
                quant_error = old_pixel - new_pixel

                for (dr, dc), factor in error_dist:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        img_array[nr, nc] = np.clip(img_array[nr, nc] + quant_error * factor, 0, 1)

        if len(img_array) > 0:
            self.image = Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')
        else:
            print("WARNING: img array in error-diffusion dithering has size 0")
        
        return self


    def halftone_dither(self, channel='r', grid_size=10, dot_scale=1.5, background_color=(255, 255, 255), dot_color=(0, 0, 0)):
        """
        Creates a halftone effect where dot size is based on color channel intensity.
        Darker areas in the channel result in larger dots.

        Args:
            channel (str): The color channel to use for intensity ('r', 'g', or 'b').
            grid_size (int): The spacing between dots in the grid.
            dot_scale (float): A multiplier for the dot size.
            background_color (tuple): RGB tuple for the background.
            dot_color (tuple): RGB tuple for the dots.
        """
        width, height = self.image.size
        
        # Create a new blank image for the output
        output_image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(output_image)

        channel_map = {'r': 0, 'g': 1, 'b': 2}
        if channel not in channel_map:
            raise ValueError("Channel must be 'r', 'g', or 'b'.")
        channel_idx = channel_map[channel]

        # Iterate over the image in a grid
        tq1 = tqdm(range(0, height, grid_size))
        tq2 = tqdm(range(0, width, grid_size), leave=False)
        tq1.set_description_str("Halftone dithering")
        for y in tq1:
            for x in tq2:
                # Define the box to average the color from
                box = (x, y, x + grid_size, y + grid_size)
                region = self.image.crop(box)
                
                # Calculate the average intensity of the chosen channel in the region
                # We use np.mean for efficiency
                region_array = np.array(region)
                # Check if region is not empty
                if region_array.size == 0:
                    continue
                    
                avg_intensity = np.mean(region_array[:, :, channel_idx])
                
                # Invert intensity because we want darker areas to have bigger dots
                # (0 intensity = max radius, 255 intensity = 0 radius)
                normalized_intensity = (255 - avg_intensity) / 255.0
                
                # Calculate dot radius
                max_radius = (grid_size / 2) * dot_scale
                radius = max_radius * normalized_intensity
                
                # Don't draw if radius is too small
                if radius < 0.1:
                    continue

                # Calculate bounding box for the circle
                dot_x = x + grid_size / 2
                dot_y = y + grid_size / 2
                bbox = [dot_x - radius, dot_y - radius, dot_x + radius, dot_y + radius]
                
                draw.ellipse(bbox, fill=dot_color)

        self.image = output_image
        return self


    def density_halftone(self, channel='r', num_dots=1e6, dot_size=1, background_color=(255, 255, 255), dot_color=(0, 0, 0)):
        """
        Creates a halftone effect where dot DENSITY is based on color channel intensity.
        Darker areas in the channel result in a higher density of dots. All dots are the same size.
        This is also known as stochastic dithering.

        Args:
            channel (str): The color channel to use for intensity ('r', 'g', or 'b').
            num_dots (int): The total number of dots to attempt to place on the image.
            dot_size (int): The radius of each individual dot.
            background_color (tuple): RGB tuple for the background.
            dot_color (tuple): RGB tuple for the dots.

        """

        width, height = self.image.size
        img_array = np.array(self.image)
        
        # Create a new blank image for the output
        output_image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(output_image)

        channel_map = {'r': 0, 'g': 1, 'b': 2}
        if channel not in channel_map:
            raise ValueError("Channel must be 'r', 'g', or 'b'.")
        channel_idx = channel_map[channel]

        
        # XXX dot color?

        # dot color is the average of the channels
        dot_color = (int(np.average(img_array[0])), int(np.average(img_array[1])), int(np.average(img_array[2])))
            

        # Perform n attempts to place a dot
        tq = tqdm(range(int(num_dots)))
        tq.set_description(f"generating density halftone with num_dots={int(num_dots)}")
        for _ in tq:
            # Pick a random coordinate
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            # Get the intensity of the chosen channel at that pixel
            intensity = img_array[y, x, channel_idx]
            
            # Invert intensity and normalize to get a probability (0-1)
            # Darker pixels (lower intensity) should have a higher probability of getting a dot.
            probability = (255 - intensity) / 255.0

            # If our random roll is less than the probability, we draw a dot
            if np.random.random() < probability:
                dot_color = (img_array[y, x, 0], img_array[y, x, 1], img_array[y, x, 2])
                # Calculate bounding box for the circle
                bbox = [x - dot_size, y - dot_size, x + dot_size, y + dot_size]
                draw.ellipse(bbox, fill=dot_color)

        self.image = output_image
        return self


    def make_indexed_png(self):
        """
        Makes the image an indexed PNG with a custom palette.
        This is the key to reducing file size.

        """
        # Create a new palette image for quantization
        palette_img = Image.new("P", (1, 1))
        # Flatten the palette and convert to integer
        palette_flat = (self.image_palette_normalized.flatten() * 255).astype(np.uint8).tolist()
        palette_img.putpalette(palette_flat)

        # Quantize the dithered image to the new palette
        # This converts the image to 'P' (Palette) mode.
        self.image = self.image.quantize(palette=palette_img, dither=Image.NONE)
        return self


    def compress_for_web(self, num_colors=16):
        """
        Shorthand for the Web Dithering pipeline.\n
        Compresses an image by generating an optimal palette and applying dithering.

        Args:
                num_colors (int): The number of colors in the final palette.
                                                Fewer colors mean smaller file size. 16, 32, or 64 are good values.

        """

        return (self
            .generate_optimized_palette(num_colors)
            .floyd_steinberg()
            .make_indexed_png()
        )

