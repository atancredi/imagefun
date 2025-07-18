from PIL import Image, ImageEnhance

Enhancer = ImageEnhance.Brightness | ImageEnhance.Contrast | ImageEnhance.Color | ImageEnhance.Sharpness

def image_enhance(im: Image.Image, enhancer: Enhancer, value):
    return enhancer(im).enhance(value)