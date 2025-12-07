import cv2 as cv

class Preprocessor:
    def resize(self, image, width=500):
        (h, w) = image.shape[:2]
        r = width / float(w)
        dim = (width, int(h * r))
        resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        return resized, r

    # Logic: Canny + Light Blur + Light Dilation
    # Goal: Get the perfect outline without bloating.
    def pipeline_precision(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 30, 100)


        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        edges = cv.dilate(edges, kernel, iterations=1)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        return edges


    # Logic: Adaptive Thresh + Median Blur
    # Goal: See edges that Canny missed.
    def pipeline_sensitivity(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (9,9), 0) 
        
        thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv.THRESH_BINARY, 91, 2)
        thresh = cv.bitwise_not(thresh)
        
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        edges = cv.dilate(thresh, kernel, iterations=1)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        return edges
    
    # Logic: Adaptive + Heavy Dilation
    # Goal: Glue broken lines together no matter what.
    def pipeline_brute_force(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.medianBlur(gray, 11)
        
        thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv.THRESH_BINARY, 21, 2)
        thresh = cv.bitwise_not(thresh)
        

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        edges = cv.dilate(thresh, kernel, iterations=3)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        return edges