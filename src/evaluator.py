import json
import numpy as np
from shapely.geometry import Polygon

class Evaluator:
    def load_ground_truth(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check if the 'shapes' list exists and is not empty
        # 'shapes' is a LIST of dictionaries: [ {label:..., points:...}, {...} ]
        if not data.get('shapes'):
            print(f"Warning: No valid shapes found in {json_path}")
            # FIX: Added [] inside np.array to fix SyntaxError
            return np.array([], dtype="float32")
            
        try:
            # FIX: Access the FIRST element [0] of the list before asking for 'points'
            points = data['shapes'][0]['points']
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not extract points from {json_path}: {e}")
            return np.array([], dtype="float32")
            
        return np.array(points, dtype="float32")

    def calculate_iou(self, gt_points, pred_points):
        if len(gt_points) == 0 or len(pred_points) == 0:
            return 0.0

        try:
            # Create polygons
            poly_gt = Polygon(gt_points)
            poly_pred = Polygon(pred_points)

            # Fix invalid geometries (e.g., self-intersecting lines)
            if not poly_gt.is_valid:
                poly_gt = poly_gt.buffer(0)
            if not poly_pred.is_valid:
                poly_pred = poly_pred.buffer(0)

            # Calculate IoU
            intersection = poly_gt.intersection(poly_pred).area
            union = poly_gt.union(poly_pred).area

            if union == 0:
                return 0.0

            return intersection / union
        except Exception as e:
            print(f"IoU Error: {e}")
            return 0.0