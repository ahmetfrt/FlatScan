import cv2 as cv
import os
import glob
from src.preprocessor import Preprocessor
from src.detector import Detector
from src.warper import Warper
from src.evaluator import Evaluator

# Initializing classes
prep = Preprocessor()
detector = Detector()
warper = Warper()
evaluator = Evaluator()

# Path to data
image_paths = glob.glob("dataset/*.jpg")

for img_path in image_paths:

    # Loading the image
    image = cv.imread(img_path)
    if image is None: continue

    # Resizing(for speed) and keeping ratio
    resized_image, ratio = prep.resize(image,width=500)
    cv.imshow("Resized", resized_image)

    # Preprocessing
    blurred = prep.apply_blur(resized_image,method="Gaussian")
    cv.imshow("Blurred", blurred)

    edges = prep.detect_edges(blurred)
    cv.imshow("Edges", edges)

    # Detecting contours
    cnts = detector.findContours(edges)
    doc_cnt = detector.get_document_contour(cnts)

    if doc_cnt is not None:
        # Scaling the detected contour back to the original image size
        doc_cnt_original = doc_cnt.reshape(4, 2) * (1 / ratio)

        # Warping the image
        warped = warper.four_point_transform(image, doc_cnt_original)
        cv.imshow("Warped", warped)

        # Evaluating
        json_path = img_path.replace(".jpg", ".json")
        if os.path.exists(json_path):
            gt_points = evaluator.load_ground_truth(json_path)
            iou_score = evaluator.calculate_iou(gt_points,doc_cnt_original)
            print(f"Image: {img_path} | IoU: {iou_score:.4f}")

        # Showing the results
        cv.imshow("Original", cv.resize(image, (500, 700)))
        cv.imshow("Warped", cv.resize(warped, (500, 700)))
        cv.waitKey(0)

    else:
        print(f"Failed to detect document in {img_path}")

cv.destroyAllWindows()