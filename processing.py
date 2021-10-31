import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from skimage.filters import threshold_local

def get_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Filter noise
    gray = cv2.medianBlur(gray,3)
    # Get clean binary image
    binary = threshold_local(gray, 21, method = "gaussian", offset = 10)
    sketch = (gray < binary).astype("uint8")
    # Dilate contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    sketch = cv2.dilate(sketch,kernel)
    return sketch 

def putText(image, text, xy, color=(255,255,255), size=10):
    img_pil = Image.fromarray(image) # cv2 to PIL
    draw = ImageDraw.Draw(img_pil)
    draw.text(xy,text,color,ImageFont.truetype("arial.ttf", size))
    return np.asarray(img_pil) # PIL to cv2
