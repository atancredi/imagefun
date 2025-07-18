from PIL import Image

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