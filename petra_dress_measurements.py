import cv2
import torch
import numpy as np
import json
from datetime import datetime
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.model_zoo import model_zoo
    from detectron2.data import MetadataCatalog
except ImportError as e:
    raise ImportError("detectron2 is not installed. Please install it using 'pip install detectron2' or follow the setup instructions from https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md") from e

class PetraDressMeasurements:
    """PETRA Dress Measurement System following PETRA_INQ.xlsm specifications"""
    
    def __init__(self, image_path="dress_wire.jpeg"):
        self.image_path = image_path
        self.image = None
        self.contour = None
        self.measurements = {}
        self.size_standards = {
            '50': {'A': 19, 'B': 52, 'C': 10, 'D': 9.5, 'E': 29, 'F': 15, 'G': 1.25, 'H': 20.5, 'I': 8, 'J': 6.5, 'K': 10, 'L': 4.5, 'M': 1.5, 'N': 46},
            '56': {'A': 20.5, 'B': 56, 'C': 10.5, 'D': 10.75, 'E': 32, 'F': 16, 'G': 1.25, 'H': 22.5, 'I': 8.5, 'J': 6.5, 'K': 10, 'L': 4.5, 'M': 1.5, 'N': 46},
            '62': {'A': 22, 'B': 60, 'C': 11, 'D': 12.5, 'E': 36, 'F': 17, 'G': 1.5, 'H': 24.5, 'I': 9, 'J': 6.5, 'K': 11, 'L': 5, 'M': 1.5, 'N': 48},
            '68': {'A': 23, 'B': 65, 'C': 11.5, 'D': 14.25, 'E': 40, 'F': 18, 'G': 1.5, 'H': 26.25, 'I': 9.5, 'J': 6.5, 'K': 11, 'L': 5, 'M': 1.5, 'N': 48},
            '74': {'A': 24, 'B': 70, 'C': 12, 'D': 16, 'E': 44, 'F': 19, 'G': 1.5, 'H': 28, 'I': 10, 'J': 7, 'K': 12, 'L': 5.5, 'M': 2, 'N': 50},
            '80': {'A': 25, 'B': 75, 'C': 12.5, 'D': 17.75, 'E': 48, 'F': 20, 'G': 1.5, 'H': 30, 'I': 10.5, 'J': 7, 'K': 12, 'L': 5.5, 'M': 2, 'N': 50}
        }
        
        self.measurement_descriptions = {
            'A': '½ CHEST',
            'B': '½ BOTTOM', 
            'C': 'ARMHOLE DEPTH fr HPS',
            'D': 'TO SKIRT AT SIDE',
            'E': 'BACK LENGTH fr HPS',
            'F': 'SHOULDER TO SHOULDER',
            'G': 'SLANTING SHOULDER',
            'H': 'SLEEVE LENGTH',
            'I': 'BICEPS',
            'J': 'BOTTOM SLEEVE',
            'K': 'NECKWIDTH',
            'L': 'NECKDROP CF fr HPS',
            'M': 'NECKDROP CB fr HPS',
            'N': 'Min NECKLINE extended'
        }
        
    def load_and_process_image(self):
        """Load image and detect dress contour using detectron2"""
        print("Loading and processing image...")
        
        # Load image
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        orig = self.image.copy()
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Load Mask R-CNN model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        predictor = DefaultPredictor(cfg)
        outputs = predictor(image_rgb)
        
        # Extract masks
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy()
        
        print(f"Number of instances detected: {len(masks)}")
        if len(masks) > 0:
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            metadata = MetadataCatalog.get("coco_2017_val")
            class_names = metadata.thing_classes
            print("Detected objects:")
            for i, (score, class_id) in enumerate(zip(scores, classes)):
                print(f"  {i+1}. {class_names[class_id]} (confidence: {score:.3f})")
        
        if len(masks) == 0:
            raise ValueError("No instances detected in the image.")
        
        # Combine masks and find contour
        combined_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask.astype(np.uint8) * 255)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            self.contour = max(contours, key=cv2.contourArea)
            self.contour = self.contour.squeeze()
            print(f"Found contour with {len(self.contour)} points")
        else:
            raise ValueError("No contours found in the mask.")
        
        return orig
    
    def find_closest_point_index(self, contour, point):
        """Find the closest point on contour to given point"""
        point = np.array(point)
        dists = np.sum((contour - point) ** 2, axis=1)
        return np.argmin(dists)
    
    def calculate_arc_length(self, segment):
        """Calculate arc length of a contour segment"""
        if len(segment) < 2:
            return 0.0
        diffs = np.diff(segment, axis=0)
        return np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
    
    def calculate_measurement(self, point1, point2, measurement_type="linear"):
        """Calculate measurement between two points"""
        if measurement_type == "linear":
            return np.linalg.norm(np.array(point1) - np.array(point2))
        elif measurement_type == "arc":
            # For arc measurements along the contour
            i1 = self.find_closest_point_index(self.contour, point1)
            i2 = self.find_closest_point_index(self.contour, point2)
            i_min, i_max = min(i1, i2), max(i1, i2)
            
            segment1 = self.contour[i_min:i_max + 1]
            len1 = self.calculate_arc_length(segment1)
            segment2 = np.vstack((self.contour[i_max:], self.contour[:i_min + 1]))
            len2 = self.calculate_arc_length(segment2)
            
            return min(len1, len2)  # Return shorter path
    
    def determine_dress_size(self, measurements):
        """Determine the closest dress size based on measurements"""
        best_size = None
        min_error = float('inf')
        
        for size, standards in self.size_standards.items():
            total_error = 0
            for measurement_id, measured_value in measurements.items():
                if measurement_id in standards:
                    standard_value = standards[measurement_id]
                    # Convert to same units (assuming measurements are in pixels, convert to cm)
                    # This is a rough conversion - in practice you'd need proper calibration
                    error = abs(measured_value - standard_value)
                    total_error += error
            
            if total_error < min_error:
                min_error = total_error
                best_size = size
        
        return best_size, min_error
    
    def draw_measurement_line(self, img, point1, point2, label, measurement_value, color=(255, 0, 0), thickness=2):
        """Draw measurement line with label and value"""
        # Draw line
        cv2.line(img, point1, point2, color, thickness)
        
        # Calculate midpoint for label
        mid_x = (point1[0] + point2[0]) // 2
        mid_y = (point1[1] + point2[1]) // 2
        
        # Draw label with measurement value
        label_text = f"{label}: {measurement_value:.1f}cm"
        cv2.putText(img, label_text, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, label_text, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw measurement points
        cv2.circle(img, point1, 5, color, -1)
        cv2.circle(img, point2, 5, color, -1)
    
    def calculate_petra_measurements(self):
        """Calculate all PETRA measurements according to specifications"""
        if self.contour is None:
            raise ValueError("No contour available. Run load_and_process_image() first.")
        
        height, width = self.image.shape[:2]
        
        # Define measurement points based on dress proportions
        # These are approximate positions that would need to be refined based on actual dress detection
        measurement_points = {
            'A': ((int(width * 0.3), int(height * 0.4)), (int(width * 0.7), int(height * 0.4))),  # ½ CHEST
            'B': ((int(width * 0.2), int(height * 0.9)), (int(width * 0.8), int(height * 0.9))),  # ½ BOTTOM
            'C': ((int(width * 0.5), int(height * 0.2)), (int(width * 0.5), int(height * 0.35))),  # ARMHOLE DEPTH
            'D': ((int(width * 0.3), int(height * 0.4)), (int(width * 0.3), int(height * 0.7))),  # TO SKIRT AT SIDE
            'E': ((int(width * 0.5), int(height * 0.1)), (int(width * 0.5), int(height * 0.9))),  # BACK LENGTH
            'F': ((int(width * 0.2), int(height * 0.2)), (int(width * 0.8), int(height * 0.2))),  # SHOULDER TO SHOULDER
            'G': ((int(width * 0.2), int(height * 0.2)), (int(width * 0.3), int(height * 0.3))),  # SLANTING SHOULDER
            'H': ((int(width * 0.2), int(height * 0.2)), (int(width * 0.1), int(height * 0.6))),  # SLEEVE LENGTH
            'I': ((int(width * 0.1), int(height * 0.4)), (int(width * 0.1), int(height * 0.5))),  # BICEPS
            'J': ((int(width * 0.1), int(height * 0.6)), (int(width * 0.1), int(height * 0.65))),  # BOTTOM SLEEVE
            'K': ((int(width * 0.4), int(height * 0.15)), (int(width * 0.6), int(height * 0.15))),  # NECKWIDTH
            'L': ((int(width * 0.5), int(height * 0.1)), (int(width * 0.5), int(height * 0.2))),  # NECKDROP CF
            'M': ((int(width * 0.5), int(height * 0.1)), (int(width * 0.5), int(height * 0.15))),  # NECKDROP CB
            'N': ((int(width * 0.3), int(height * 0.15)), (int(width * 0.7), int(height * 0.15)))  # NECKLINE extended
        }
        
        # Snap points to contour and calculate measurements
        calculated_measurements = {}
        annotated_image = self.image.copy()
        
        for measurement_id, (point1, point2) in measurement_points.items():
            # Snap to contour
            idx1 = self.find_closest_point_index(self.contour, point1)
            idx2 = self.find_closest_point_index(self.contour, point2)
            snapped_point1 = tuple(self.contour[idx1])
            snapped_point2 = tuple(self.contour[idx2])
            
            # Calculate measurement (convert pixels to cm - rough approximation)
            pixel_distance = self.calculate_measurement(snapped_point1, snapped_point2)
            # Rough conversion: assuming image represents a dress of ~50cm width
            cm_distance = (pixel_distance / width) * 50
            
            calculated_measurements[measurement_id] = cm_distance
            
            # Draw measurement line
            self.draw_measurement_line(annotated_image, snapped_point1, snapped_point2, 
                                     measurement_id, cm_distance)
        
        self.measurements = calculated_measurements
        return annotated_image, calculated_measurements
    
    def generate_measurement_report(self):
        """Generate a professional measurement report"""
        if not self.measurements:
            raise ValueError("No measurements available. Run calculate_petra_measurements() first.")
        
        # Determine dress size
        determined_size, error = self.determine_dress_size(self.measurements)
        
        report = {
            "item_info": {
                "item_name": "PETRA",
                "pattern_no": "J8518-1",
                "pattern_maker": "perbra",
                "measurement_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "determined_size": determined_size,
                "measurement_error": error
            },
            "measurements": {}
        }
        
        # Add measurements with descriptions
        for measurement_id, value in self.measurements.items():
            report["measurements"][measurement_id] = {
                "description": self.measurement_descriptions[measurement_id],
                "measured_value": round(value, 2),
                "unit": "cm"
            }
        
        return report
    
    def save_results(self, annotated_image, report):
        """Save annotated image and measurement report"""
        # Save annotated image
        cv2.imwrite("petra_annotated_dress.png", annotated_image)
        print("✅ Annotated image saved as 'petra_annotated_dress.png'")
        
        # Save measurement report
        with open("petra_measurement_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("✅ Measurement report saved as 'petra_measurement_report.json'")
        
        # Print summary
        print(f"\n=== PETRA DRESS MEASUREMENT SUMMARY ===")
        print(f"Determined Size: {report['item_info']['determined_size']}")
        print(f"Measurement Error: {report['item_info']['measurement_error']:.2f}")
        print(f"\nMeasurements:")
        for measurement_id, data in report["measurements"].items():
            print(f"  {measurement_id}: {data['description']} = {data['measured_value']}cm")

def main():
    """Main function to run PETRA dress measurements"""
    try:
        # Initialize measurement system
        petra = PetraDressMeasurements()
        
        # Load and process image
        original_image = petra.load_and_process_image()
        
        # Calculate measurements
        annotated_image, measurements = petra.calculate_petra_measurements()
        
        # Generate report
        report = petra.generate_measurement_report()
        
        # Save results
        petra.save_results(annotated_image, report)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
