import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
from src.preprocessor import Preprocessor
from src.detector import Detector
from src.warper import Warper
from src.evaluator import Evaluator
from src.postprocessor import PostProcessor

prep = Preprocessor()
detector = Detector()
warper = Warper()
evaluator = Evaluator()
postproc = PostProcessor()


image_paths = glob.glob("dataset/*.jpg")

# Defined to not detect too small objects as documents
SAFE_AREA = 120000 

all_ious = []
method_counts = {"Precision" : 0, "Sensitivity" : 0, "BruteForce" : 0}

for img_path in image_paths:
    filename = os.path.basename(img_path)
    image = cv.imread(img_path)
    if image is None: continue

    # resized to a standard(important for speed)
    resized_image, ratio = prep.resize(image, width=500)
    
    # defined for pipelining
    best_cnt = None
    best_method = "None"
    max_area = 0
    
    # first pipeline -> precision
    edges = prep.pipeline_precision(resized_image)
    cnts = detector.findContours(edges)
    cnt = detector.get_document_contour(cnts)
    
    if cnt is not None:
        best_cnt = cnt
        best_method = "Precision"
        max_area = cv.contourArea(cnt)


    if cnt is None or max_area < 40000:
    # second pipeline -> sensitivity
        if max_area < 80000:
            edges = prep.pipeline_sensitivity(resized_image)
            cnts = detector.findContours(edges)
            cnt = detector.get_document_contour(cnts)
            
            if cnt is not None:
                area = cv.contourArea(cnt)
                # incumbent rule -> only switch if the new area is > 30% larger than the old one
                if area > max_area * 1.3:
                    best_cnt = cnt
                    best_method = "Sensitivity"
                    max_area = area

        # third pipeline -> brute force
        if max_area < 120000:
            edges = prep.pipeline_brute_force(resized_image)
            cnts = detector.findContours(edges)
            cnt = detector.get_document_contour(cnts)
            
            if cnt is not None:
                area = cv.contourArea(cnt)
                if area > max_area * 1.3:
                    best_cnt = cnt
                    best_method = "BruteForce"
                    max_area = area

    # final processing
    doc_cnt = best_cnt
    
    debug_vis = resized_image.copy()
    if doc_cnt is not None:
        cv.drawContours(debug_vis, [doc_cnt], -1, (0, 255, 0), 3)
        
        # points resized to original
        doc_cnt_original = doc_cnt.reshape(4, 2) * (1 / ratio)

        # warp the image
        warped = warper.four_point_transform(image, doc_cnt_original)

        # to give scanned pdf look
        scanned = postproc.process(warped, mode="grayscale")

        # save the scanned version
        cv.imwrite(os.path.join("scanned", f"scanned_{filename}"), scanned)
        
        # calculate and prind IoU and used method
        json_path = img_path.replace(".jpg", ".json")
        if os.path.exists(json_path):
            gt_points = evaluator.load_ground_truth(json_path)
            iou = evaluator.calculate_iou(gt_points, doc_cnt_original)
            print(f"Image: {filename} | Used: {best_method:17} | IoU: {iou:.4f}")
            # COLLECT DATA HERE
            all_ious.append(iou)
            if best_method in method_counts:
                method_counts[best_method] += 1
        elif "Precision" in best_method: # Handle "Precision (Small)" case
            method_counts["Precision"] += 1
        else:
            print(f"Image: {filename} | Used: {best_method:17} | Detected")
        
        # save the warped version
        cv.imwrite(os.path.join("output", f"warped_{filename}"), warped)
    else:
        print(f"Failed to detect document in {filename}")

    # save the version where the contours drawn
    cv.imwrite(os.path.join("visual", f"debug_{filename}"), debug_vis)



# 2. GENERATE IoU HISTOGRAM
plt.figure(figsize=(10, 6))
plt.hist(all_ious, bins=[0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0], 
         color='green', edgecolor='black', alpha=0.7)
plt.title('Distribution of IoU Scores (Accuracy)')
plt.xlabel('Intersection over Union (IoU)')
plt.ylabel('Number of Images')
plt.grid(axis='y', alpha=0.5)

# Save the plot
plt.savefig(os.path.join("graphs", "iou_histogram.png"))
print("Saved IoU Histogram.")

# 3. GENERATE METHOD USAGE BAR CHART
plt.figure(figsize=(8, 6))
methods = list(method_counts.keys())
counts = list(method_counts.values())
bars = plt.bar(methods, counts, color=['blue', 'orange', 'red'])

# Add labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), 
             ha='center', va='bottom', fontweight='bold')

plt.title('Pipeline Stage Activation Frequency')
plt.xlabel('Detection Pipeline Stage')
plt.ylabel('Count of Images Processed')
plt.grid(axis='y', alpha=0.3)

# Save the plot
plt.savefig(os.path.join("graphs", "method_usage_chart.png"))
print("Saved Method Usage Chart.")