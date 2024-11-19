import os
from PIL import Image
import numpy as np

# Grayscaling and normalization (0-1)
def gray_norm(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('jpg')):
                image_path = os.path.join(subdir, file)
                with Image.open(image_path) as img:
                    # Grayscaling
                    gray_img = img.convert("L")
                    # Convert the image to a numpy array
                    img_array = np.array(gray_img)  
                    # Normalize the pixel values to a 0-1 range by dividing by 255
                    img_array_normalized = img_array / 255.0
                    normalized_img = Image.fromarray((img_array_normalized * 255).astype(np.uint8))
                    normalized_img.save(image_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
gray_norm(current_dir)