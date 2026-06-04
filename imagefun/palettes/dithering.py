import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from numba import njit

import warnings
warnings.filterwarnings('ignore') 

def get_bayer_matrix(size):
    if size == 2:
        return np.array([[0, 2], [3, 1]])
    else:
        sub_matrix = get_bayer_matrix(size // 2)
        return np.block([
            [4 * sub_matrix,     4 * sub_matrix + 2],
            [4 * sub_matrix + 3, 4 * sub_matrix + 1]
        ])


def find_closest_palette_color(pixel, palette: np.ndarray):
    """
    (static method)\n
    Find the closest color in the palette to a given pixel color.\n
    Uses Euclidean distance in RGB space.
    """
    distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
    return palette[np.argmin(distances)]


def find_closest_palette_color_index(pixel, palette: np.ndarray):
    """
    (static method)\n
    Find the closest color in the palette to a given pixel color and return the index in the palette.\n
    Uses Euclidean distance in RGB space.
    """
    distances = np.sqrt(np.sum((palette - pixel) ** 2, axis=1))
    i = np.argmin(distances)
    return i, palette[i]


def bayer(image: Image.Image, palette_norm, bayer_size=4):
    
    width, height = image.size
    # Ordered dithering with a Bayer matrix
    if bayer_size not in [2, 4, 8]:
        raise ValueError("Bayer matrix size must be 2, 4, or 8.")
    
    bayer_matrix = get_bayer_matrix(bayer_size)
    bayer_norm = bayer_matrix / (bayer_size**2) - 0.5

    image = image.convert("RGB")
    img_array = np.array(image, dtype=float) / 255
    
    tq1 = tqdm(range(height))
    tq1.set_description_str("bayer dithering")
    tq2 = tqdm(range(width), leave=False)
    for r in tq1:
        for c in tq2:
            threshold = bayer_norm[r % bayer_size, c % bayer_size]
            new_pixel = np.clip(img_array[r, c] + threshold, 0, 1)
            img_array[r, c] = find_closest_palette_color(new_pixel, palette_norm)
    
    image = Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')
    return image


def floyd_steinberg(image: Image.Image, palette_norm):
    # Error-diffusion dithering
    image = image.convert("RGB")
    img_array = np.array(image, dtype=float) / 255
    width, height = image.size

    error_dist = [((0, 1), 7/16), ((1, -1), 3/16), ((1, 0), 5/16), ((1, 1), 1/16)]

    tq1 = tqdm(range(height))
    tq1.set_description_str("floyd steinberg dithering")
    tq2 = tqdm(range(width), leave=False)
    for r in tq1:
        for c in tq2:
            old_pixel = img_array[r, c].copy()
            new_pixel = find_closest_palette_color(old_pixel, palette_norm)
            img_array[r, c] = new_pixel
            
            quant_error = old_pixel - new_pixel

            for (dr, dc), factor in error_dist:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    img_array[nr, nc] = np.clip(img_array[nr, nc] + quant_error * factor, 0, 1)

    image = Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')
    return image



# @njit compiles this function to machine code. fastmath=True allows further CPU optimizations.
@njit(fastmath=True)
def _atkinson_dither_core(img_array, palette_norm):
    height, width, _ = img_array.shape
    num_colors = palette_norm.shape[0]

    for r in range(height):
        for c in range(width):
            # 1. Read pixel directly (no .copy() array allocation)
            p0 = img_array[r, c, 0]
            p1 = img_array[r, c, 1]
            p2 = img_array[r, c, 2]

            # 2. Find closest palette color (Inlined Euclidean distance for speed)
            min_dist = 1e8
            best_idx = 0
            for i in range(num_colors):
                d0 = p0 - palette_norm[i, 0]
                d1 = p1 - palette_norm[i, 1]
                d2 = p2 - palette_norm[i, 2]
                
                # Squared distance is faster than calculating the square root
                dist = d0*d0 + d1*d1 + d2*d2 
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

            # 3. Assign new pixel
            new_p0 = palette_norm[best_idx, 0]
            new_p1 = palette_norm[best_idx, 1]
            new_p2 = palette_norm[best_idx, 2]

            img_array[r, c, 0] = new_p0
            img_array[r, c, 1] = new_p1
            img_array[r, c, 2] = new_p2

            # 4. Calculate quantization error (Multiply by 0.125 instead of dividing by 8)
            err0 = (p0 - new_p0) * 0.125
            err1 = (p1 - new_p1) * 0.125
            err2 = (p2 - new_p2) * 0.125

            # 5. Distribute error (Unrolled loops and manual clipping for maximum speed)
            if c + 1 < width:
                v0 = img_array[r, c + 1, 0] + err0; img_array[r, c + 1, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                v1 = img_array[r, c + 1, 1] + err1; img_array[r, c + 1, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                v2 = img_array[r, c + 1, 2] + err2; img_array[r, c + 1, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)
            
            if c + 2 < width:
                v0 = img_array[r, c + 2, 0] + err0; img_array[r, c + 2, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                v1 = img_array[r, c + 2, 1] + err1; img_array[r, c + 2, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                v2 = img_array[r, c + 2, 2] + err2; img_array[r, c + 2, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)
            
            if r + 1 < height:
                if c - 1 >= 0:
                    v0 = img_array[r + 1, c - 1, 0] + err0; img_array[r + 1, c - 1, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                    v1 = img_array[r + 1, c - 1, 1] + err1; img_array[r + 1, c - 1, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                    v2 = img_array[r + 1, c - 1, 2] + err2; img_array[r + 1, c - 1, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)
                
                v0 = img_array[r + 1, c, 0] + err0; img_array[r + 1, c, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                v1 = img_array[r + 1, c, 1] + err1; img_array[r + 1, c, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                v2 = img_array[r + 1, c, 2] + err2; img_array[r + 1, c, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)
                
                if c + 1 < width:
                    v0 = img_array[r + 1, c + 1, 0] + err0; img_array[r + 1, c + 1, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                    v1 = img_array[r + 1, c + 1, 1] + err1; img_array[r + 1, c + 1, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                    v2 = img_array[r + 1, c + 1, 2] + err2; img_array[r + 1, c + 1, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)
            
            if r + 2 < height:
                v0 = img_array[r + 2, c, 0] + err0; img_array[r + 2, c, 0] = 0.0 if v0 < 0.0 else (1.0 if v0 > 1.0 else v0)
                v1 = img_array[r + 2, c, 1] + err1; img_array[r + 2, c, 1] = 0.0 if v1 < 0.0 else (1.0 if v1 > 1.0 else v1)
                v2 = img_array[r + 2, c, 2] + err2; img_array[r + 2, c, 2] = 0.0 if v2 < 0.0 else (1.0 if v2 > 1.0 else v2)
                
    return img_array

def diffusion(image: Image.Image, palette_norm):
    # check for numba
    palette_norm = np.ascontiguousarray(palette_norm, dtype=np.float64)
    
    image = image.convert("RGB")
    img_array = np.array(image, dtype=np.float64) / 255.0
    
    print("Applying Atkinson error diffusion...")
    img_array = _atkinson_dither_core(img_array, palette_norm)
    
    return Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')

def halftone_dither(
        image: Image.Image,
        channel='r',
        grid_size=10,
        dot_scale=1.5,
        background_color=(255, 255, 255),
        dot_color=(0, 0, 0)
    ) -> Image.Image:
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
        width, height = image.size
        
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
                region = image.crop(box)
                
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

        return output_image


def density_halftone(
        image: Image.Image,
        channel='r',
        num_dots=1e6,
        dot_size=1,
        background_color=(255, 255, 255),
        dot_color=(0, 0, 0)
    ) -> Image.Image:
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

    width, height = image.size
    img_array = np.array(image)
    
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

    return output_image

