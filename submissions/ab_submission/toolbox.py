
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

font_size = 60
if os.name != 'nt':
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", font_size)
else:
    font = ImageFont.truetype("C:/Windows/Fonts/Arial/arialbd.ttf", font_size)

def write_rectangle(image, preds, folder=None, filename=None, negative=True):
    img = Image.fromarray(image)
    img_draw = ImageDraw.Draw(img)
    for pred in preds:
        write = True
        if negative == False and pred["class"] == "Negative":
            write = False

        if write:
            x1, y1, x2, y2 = pred["bbox"]
            label = pred["class"]
            img_draw.rounded_rectangle(((x1, y1), (x2,y2)), fill=None, outline="black", width=5)
            img_draw.text((x1, y1-70), label, font=font, fill="black")

    if folder is not None and filename is not None:
        img.save(f"{folder}/{filename}")
    
    return np.array(img)