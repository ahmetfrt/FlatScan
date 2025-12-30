import cv2 as cv
import numpy as np

class PostProcessor:
    def __init__(self):
        pass

    def process(self, image):
        # 1. Convert to Grayscale
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        
        # Dilate creates a "max" filter, removing thin dark text
        dilated = cv.dilate(gray, np.ones((7, 7), np.uint8))
        
        # Heavy blur to smooth out the background estimate
        bg_shadow = cv.medianBlur(dilated, 21)
        
        diff_img = 255 - cv.absdiff(gray, bg_shadow)
        
        # Normalize to stretch the contrast to full 0-255 range
        norm_img = cv.normalize(diff_img, None, alpha=0, beta=255, 
                                norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

        # Using OTSU thresholding as a "soft clamp" to make text darker
        _, thr = cv.threshold(norm_img, 230, 0, cv.THRESH_TRUNC)
        return cv.normalize(thr, None, 0, 255, cv.NORM_MINMAX)