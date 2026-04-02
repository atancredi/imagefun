import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

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

