import numpy as np
from PIL import Image
from tqdm import tqdm

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


def diffusion(image: Image.Image, palette_norm):
    # Error-diffusion dithering
    # DOES NOT DEPEND ON PALETTE SIZE
    image = image.convert("RGB")
    img_array = np.array(image, dtype=float) / 255
    width, height = image.size

    error_dist = [((0, 1), 1/8), ((0, 2), 1/8), ((1, -1), 1/8), 
                        ((1, 0), 1/8), ((1, 1), 1/8), ((2, 0), 1/8)] # Atkinson

    tq1 = tqdm(range(height))
    tq1.set_description_str("error diffusion dithering")
    tq2 = tqdm(range(width), leave=False)
    indexed_array = np.zeros((height, width), dtype=int)
    for r in tq1:
        for c in tq2:
            old_pixel = img_array[r, c].copy()
            i, new_pixel = find_closest_palette_color_index(old_pixel, palette_norm)
            indexed_array[r, c] = i

            # # white detection
            # if np.average(old_pixel) < 0.05:
            #     new_pixel = [1.,1.,1.]
            #     detected_whites += 1
            #     tq1.set_description_str(f"error diffusion dithering | detected white pixels: {detected_whites}")

            img_array[r, c] = new_pixel
            
            quant_error = old_pixel - new_pixel

            for (dr, dc), factor in error_dist:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    img_array[nr, nc] = np.clip(img_array[nr, nc] + quant_error * factor, 0, 1)

    image = Image.fromarray((img_array * 255).astype(np.uint8), 'RGB')
    return image

