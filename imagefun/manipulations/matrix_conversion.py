from PIL import Image, ImageFilter
import numpy as np

def get_tuple_from_matrix(matrix):
    return (matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0],
            matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1],
            matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2])

cool_matrix_01 = [
    [42,5,66],
    [0,0,0],
    [-40,-40,-40],
    [-55,-55,-55]
]

def matrix_conversion(im: Image.Image, matrix):
    return im.convert("RGB", matrix=get_tuple_from_matrix(matrix))

def edge_detect_pil(im, matrix, threshold=50, alpha=1.0):
    # Load grayscale image
    image = im.convert("L")
    
    # Define Laplacian kernel (thin edges)
    matrix = [ x for xs in matrix for x in xs ]
    
    # Apply convolution
    filtered = image.filter(ImageFilter.Kernel((3, 3), matrix, scale=1, offset=0))
    
    # Convert to numpy for normalization
    arr = np.array(filtered, dtype=float)
    
    # # Normalize to 0–255
    # arr -= arr.min()
    # if arr.max() > 0:
    #     arr = arr / arr.max() * 255
    
    # Apply threshold to thin edges
    # arr = np.where(arr > threshold, 255, 0).astype(np.uint8)
    
    # Back to Pillow image
    edges = Image.fromarray(arr, mode="L")
    
    return edges
