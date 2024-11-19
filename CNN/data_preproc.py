import os
from PIL import Image

# Grayscaling
def grayscaling(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('jpg')):
                image_path = os.path.join(subdir, file)
                with Image.open(image_path) as img:
                    grayscale_img = img.convert("L")
                    grayscale_img.save(image_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
grayscaling(current_dir)