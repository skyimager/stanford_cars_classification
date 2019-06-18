import os
import numpy as np
import base64
import io
from PIL import Image
    
def get_image(encoded_content):
    content_bytes = encoded_content.encode('utf-8')
    byte_string = base64.decodebytes(content_bytes)
    image = Image.open(io.BytesIO(byte_string))
    w,h = image.size
    num_channels=3
    byte_image = image.tobytes()
    image_arr = np.frombuffer(byte_image, dtype=np.uint8)
    image_arr = image_arr.reshape(h, w, num_channels)
    
    return image_arr


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise