import cv2 as cv
import numpy as np

# Could not understand clearly
class Warper:
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")


        # 1. The Top-Left point will have the smallest sum (x + y)
        # 2. The Bottom-Right point will have the largest sum (x + y)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-Left
        rect[2] = pts[np.argmax(s)] # Bottom-Right

        # 3. The Top-Right point will have the smallest difference (x - y)
        # 4. The Bottom-Left point will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-Right
        rect[3] = pts[np.argmax(diff)] # Bottom-Left

        return rect
    
    def four_point_transform(self,image,pts):
        rect = self.order_points(pts)

        (tl, tr, br, bl) = rect

        width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(width_A), int(width_B))


        height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(height_A), int(height_B))


        """dst = np.zeros((4,2), dtype="float32")
        dst[0] = [0,0]
        dst[1] = [0, maxHeight-1]
        dst[2] = [maxWidth-1, maxHeight-1]
        dst[3] = [maxWidth-1, 0]

        dst = dst.astype("float32")"""

        dst = np.array([
            [0, 0],                       # Top-Left
            [maxWidth - 1, 0],            # Top-Right
            [maxWidth - 1, maxHeight - 1],# Bottom-Right
            [0, maxHeight - 1]],          # Bottom-Left
            dtype="float32"
        )

        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
        


        