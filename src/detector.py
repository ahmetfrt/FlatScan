import cv2 as cv
import numpy as np

class Detector:
    def findContours(self, edge_image):
        cnts, _ = cv.findContours(edge_image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]
        return cnts
    
    def get_document_contour(self, cnts):
        for c in cnts:
            # Filter out small noise
            if cv.contourArea(c) < 2000:
                continue
            
            # FIX: Convex Hull makes the shape "solid" (like wrapping a rubber band around it)
            # This fixes issues where the edge is slightly jagged.
            c = cv.convexHull(c)

            peri = cv.arcLength(c, True)
            # 0.04 allows for slightly imperfect rectangles
            approx = cv.approxPolyDP(c, 0.04 * peri, True)

            if len(approx) == 4:
                return approx
        
        return None