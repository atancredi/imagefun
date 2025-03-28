from PIL import Image

def crop_transparent_png(img, vertical_tolerance=0):
    
    # Get the alpha channel
    alpha = img.split()[3]
    
    # Get bounding box of non-transparent areas
    bbox = alpha.getbbox()
    
    # if bbox:
    #     # Crop the image to the bounding box
    #     cropped_img = img.crop(bbox)
    #     # Save the cropped image
    #     return cropped_img
    
    if bbox:
        left, upper, right, lower = bbox

        # Apply vertical tolerance
        upper = max(0, upper + vertical_tolerance)
        lower = min(img.height, lower - vertical_tolerance)

        # Crop the image to the modified bounding box
        cropped_img = img.crop((left, upper, right, lower))

        return cropped_img
    
    return img

def add_watermark(image_path, watermark_path, position, margin_percent, size, output_path):
    # Open the main image and ensure it's square
    image = Image.open(image_path).convert("RGBA")
    img_width, img_height = image.size
    if img_width != img_height:
        print(f"The main image must be square (sizes) {img_width} x {img_height}")
        print()
        image = image.resize((img_height, img_height))

    # Open the watermark image with alpha transparency
    watermark = Image.open(watermark_path).convert("RGBA")
    
	# Crop transparency from watermark
    watermark = crop_transparent_png(watermark)

    # Define watermark sizes based on image width
    size_map = {
        "big": img_width * 0.5,  # 50% of image width
        "medium": img_width * 0.25,  # 25% of image width
        "small": img_width * 0.125  # 12.5% of image width
    }

    if size not in size_map:
        raise ValueError("Invalid size. Use 'big', 'medium', or 'small'.")

    # Resize watermark while keeping aspect ratio
    wm_new_width = int(size_map[size])
    wm_aspect_ratio = watermark.height / watermark.width
    wm_new_height = int(wm_new_width * wm_aspect_ratio)
    watermark = watermark.resize((wm_new_width, wm_new_height), Image.ANTIALIAS)

    # Compute a **single margin** based on the main image size (NOT watermark size)
    # margin = int((margin_percent / 100) * img_width)  

    margin_x = int((margin_percent / 100) * wm_new_width)
    margin_y = int((margin_percent / 100) * wm_new_height)


    # # Compute position with centered alignment
    # if position == "tr":  # Top-right
    #     wm_x = img_width - wm_new_width - margin
    #     wm_y = margin
    # elif position == "tl":  # Top-left
    #     wm_x = margin
    #     wm_y = margin
    # elif position == "br":  # Bottom-right
    #     wm_x = img_width - wm_new_width - margin
    #     wm_y = img_height - wm_new_height - margin
    # elif position == "bl":  # Bottom-left
    #     wm_x = margin
    #     wm_y = img_height - wm_new_height - margin
    # else:
    #     raise ValueError("Invalid position. Use 'tr', 'tl', 'br', or 'bl'.")

     # Calculate the position for the watermark with adjusted margins
    if position == "tr":  # Top-right
        # wm_x = img_width - wm_new_width - margin_x
        wm_x = min(img_width - wm_new_width - margin_x, img_width - wm_new_width)  # Ensure it stays in bounds
        wm_y = margin_y
    elif position == "tl":  # Top-left
        wm_x = margin_x
        wm_y = margin_y
    elif position == "br":  # Bottom-right

        
        # wm_x = img_width - wm_new_width - margin_x
        wm_x = min(img_width - wm_new_width - margin_x, img_width - wm_new_width)

        # wm_y = img_height - wm_new_height - margin_y
        wm_y = max(img_height - wm_new_height - margin_y, 0)  # Ensure no bleeding
    elif position == "bl":  # Bottom-left
        wm_x = margin_x
        # wm_y = img_height - wm_new_height - margin_y
        wm_y = max(img_height - wm_new_height - margin_y, 0)
    else:
        raise ValueError("Invalid position. Use 'tr', 'tl', 'br', or 'bl'.")


    # Adjust position to center the watermark correctly
    wm_x = max(margin_x, min(wm_x, img_width - wm_new_width - margin_x))
    wm_y = max(margin_y, min(wm_y, img_height - wm_new_height - margin_y))


    # Paste the watermark onto the image
    image.paste(watermark, (wm_x, wm_y), watermark)

    # Save the output image
    image.save(output_path, "PNG")


# Example usage
add_watermark("test_square.png", "logogifwhite.png", "tr", 10, "medium", "output.png")
