import cv2 as cv
import numpy as np

class Detector:
    def findContours(self, edge_image):
        cnts, _ = cv.findContours(edge_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]
        return cnts
    
    def get_document_contour(self, cnts):
        for c in cnts:
            # to not see small noises
            if cv.contourArea(c) < 40000:
                continue
            
            # makes the shape more solid
            c = cv.convexHull(c)

            peri = cv.arcLength(c, True)
            # was 0.02 previously but 0.04 is better for slightly imperfect rectangles
            approx = cv.approxPolyDP(c, 0.04 * peri, True)

            if len(approx) == 4:
                return approx
        
        return None