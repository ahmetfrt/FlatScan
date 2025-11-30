import cv2 as cv
import numpy as np

class Preprocessor:
    def __init__(self):
        pass

    def resize(self, image, width = 500):
        # Resizing image to a standard width(important for speed)

        # Shape of image in terms of pixels
        (h, w) = image.shape[:2]

        # Calculating ratio to apply final crop to the original image
        r = width / float(w)

        dim = (width, int(h * r))
        resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        return resized, r

    def apply_blur(self, image, method = "Bilateral"):
        # Method parameter will be used for ablation study

        # Gray color space
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Default is gaussian but both will be tested for comparison
        if method == "Gaussian":
            return cv.GaussianBlur(gray, (7,7), 0)
        elif method == "Bilateral":
            return cv.bilateralFilter(gray, 9, 75, 75)
        
        return gray
    
    def detect_edges(self, image):
        # FIX 1: Use Fixed Thresholds.
        # (30, 100) works much better for soft document edges than dynamic ones.
        lower, upper = 30, 100

        # Applying Canny Edge Detector
        edges = cv.Canny(image, lower, upper)

        # FIX 2: Stronger Dilation
        # We use iterations=2 to bridge larger gaps in the paper edge
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        edges = cv.dilate(edges, kernel, iterations=2)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

        return edges