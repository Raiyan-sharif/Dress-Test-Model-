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

class FinalDressAnnotator:
    """Final Professional Dress Annotator - Creates complete technical drawing"""
    
    def __init__(self, image_path="dress_wire.jpeg"):
        self.image_path = image_path
        self.image = None
        self.contour = None
        self.measurements = {}
        
        # PETRA measurement standards
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
    
    def calculate_measurement(self, point1, point2):
        """Calculate measurement between two points"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def draw_technical_measurement_line(self, img, point1, point2, label, measurement_value, 
                                      color=(0, 0, 0), thickness=2, offset=25):
        """Draw technical drawing style measurement line"""
        
        # Calculate line properties
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
        
        # Normalize direction vector
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Calculate perpendicular offset for measurement line
        perp_x = -dy_norm * offset
        perp_y = dx_norm * offset
        
        # Offset points for measurement line
        offset_p1 = (int(point1[0] + perp_x), int(point1[1] + perp_y))
        offset_p2 = (int(point2[0] + perp_x), int(point2[1] + perp_y))
        
        # Draw main measurement line
        cv2.line(img, offset_p1, offset_p2, color, thickness)
        
        # Draw extension lines (perpendicular to measurement line)
        ext_length = 20
        ext1_start = (int(point1[0] - dx_norm * ext_length), int(point1[1] - dy_norm * ext_length))
        ext1_end = (int(point1[0] + dx_norm * ext_length), int(point1[1] + dy_norm * ext_length))
        ext2_start = (int(point2[0] - dx_norm * ext_length), int(point2[1] - dy_norm * ext_length))
        ext2_end = (int(point2[0] + dx_norm * ext_length), int(point2[1] + dy_norm * ext_length))
        
        cv2.line(img, ext1_start, ext1_end, color, 1)
        cv2.line(img, ext2_start, ext2_end, color, 1)
        
        # Draw arrows at ends
        arrow_length = 12
        arrow_angle = 0.3
        
        # Arrow 1
        arrow1_end = (int(offset_p1[0] - dx_norm * arrow_length), 
                     int(offset_p1[1] - dy_norm * arrow_length))
        cv2.arrowedLine(img, offset_p1, arrow1_end, color, thickness, tipLength=arrow_angle)
        
        # Arrow 2
        arrow2_end = (int(offset_p2[0] + dx_norm * arrow_length), 
                     int(offset_p2[1] + dy_norm * arrow_length))
        cv2.arrowedLine(img, offset_p2, arrow2_end, color, thickness, tipLength=arrow_angle)
        
        # Draw measurement label
        mid_x = (offset_p1[0] + offset_p2[0]) // 2
        mid_y = (offset_p1[1] + offset_p2[1]) // 2
        
        # Create label text
        label_text = f"{label}"
        value_text = f"{measurement_value:.1f}cm"
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        (label_w, label_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        (value_w, value_h), _ = cv2.getTextSize(value_text, font, font_scale, font_thickness)
        
        # Draw white background for text
        padding = 8
        bg_x1 = mid_x - max(label_w, value_w) // 2 - padding
        bg_y1 = mid_y - label_h - value_h - padding
        bg_x2 = mid_x + max(label_w, value_w) // 2 + padding
        bg_y2 = mid_y + padding
        
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        
        # Draw text
        cv2.putText(img, label_text, (mid_x - label_w // 2, mid_y - value_h - 2), 
                   font, font_scale, color, font_thickness, cv2.LINE_AA)
        cv2.putText(img, value_text, (mid_x - value_w // 2, mid_y), 
                   font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    def draw_curved_measurement(self, img, point1, point2, label, measurement_value, 
                              color=(0, 0, 0), thickness=2):
        """Draw curved measurement along contour"""
        i1 = self.find_closest_point_index(self.contour, point1)
        i2 = self.find_closest_point_index(self.contour, point2)
        i_min, i_max = min(i1, i2), max(i1, i2)
        
        # Extract contour segment
        segment = self.contour[i_min:i_max + 1]
        
        if len(segment) > 1:
            # Draw curved line along contour
            cv2.polylines(img, [segment], False, color, thickness)
            
            # Add arrows at ends
            arrow_length = 15
            if len(segment) > 1:
                # Arrow at start
                p0 = tuple(segment[0])
                p1 = tuple(segment[1])
                dir_vec = np.array(p1) - np.array(p0)
                length = np.linalg.norm(dir_vec)
                if length > 0:
                    unit_dir = dir_vec / length
                    arrow_end = np.array(p0) - unit_dir * arrow_length
                    cv2.arrowedLine(img, p0, tuple(np.round(arrow_end).astype(int)), 
                                   color, thickness, tipLength=0.3)
                
                # Arrow at end
                pn1 = tuple(segment[-2])
                pn = tuple(segment[-1])
                dir_vec = np.array(pn) - np.array(pn1)
                length = np.linalg.norm(dir_vec)
                if length > 0:
                    unit_dir = dir_vec / length
                    arrow_end = np.array(pn) + unit_dir * arrow_length
                    cv2.arrowedLine(img, pn, tuple(np.round(arrow_end).astype(int)), 
                                   color, thickness, tipLength=0.3)
            
            # Add label
            if len(segment) > 0:
                mid_idx = len(segment) // 2
                mid = segment[mid_idx]
                
                label_text = f"{label}: {measurement_value:.1f}cm"
                cv2.putText(img, label_text, (mid[0], mid[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, label_text, (mid[0], mid[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    def create_final_technical_drawing(self):
        """Create final technical drawing with front and back views"""
        if self.contour is None:
            raise ValueError("No contour available. Run load_and_process_image() first.")
        
        height, width = self.image.shape[:2]
        
        # Create larger canvas for both views
        canvas_height = height * 2 + 100
        canvas_width = width + 200
        annotated = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Draw dress contour in black (front view - top)
        front_contour = self.contour.copy()
        cv2.drawContours(annotated, [front_contour.reshape(-1, 1, 2)], -1, (0, 0, 0), 2)
        
        # Draw dress contour (back view - bottom, mirrored)
        back_contour = self.contour.copy()
        back_contour[:, 0] = width - back_contour[:, 0]  # Mirror horizontally
        back_contour[:, 1] += height + 50  # Move down
        cv2.drawContours(annotated, [back_contour.reshape(-1, 1, 2)], -1, (0, 0, 0), 2)
        
        # Define measurement points for front view
        front_measurement_points = {
            'A': ((int(width * 0.3), int(height * 0.4)), (int(width * 0.7), int(height * 0.4))),  # ½ CHEST
            'B': ((int(width * 0.2), int(height * 0.9)), (int(width * 0.8), int(height * 0.9))),  # ½ BOTTOM
            'C': ((int(width * 0.5), int(height * 0.2)), (int(width * 0.5), int(height * 0.35))),  # ARMHOLE DEPTH
            'G': ((int(width * 0.2), int(height * 0.2)), (int(width * 0.3), int(height * 0.3))),  # SLANTING SHOULDER
            'H': ((int(width * 0.2), int(height * 0.2)), (int(width * 0.1), int(height * 0.6))),  # SLEEVE LENGTH
            'I': ((int(width * 0.1), int(height * 0.4)), (int(width * 0.1), int(height * 0.5))),  # BICEPS
            'J': ((int(width * 0.1), int(height * 0.6)), (int(width * 0.1), int(height * 0.65))),  # BOTTOM SLEEVE
            'K': ((int(width * 0.4), int(height * 0.15)), (int(width * 0.6), int(height * 0.15))),  # NECKWIDTH
            'L': ((int(width * 0.5), int(height * 0.1)), (int(width * 0.5), int(height * 0.2))),  # NECKDROP CF
        }
        
        # Define measurement points for back view
        back_measurement_points = {
            'D': ((int(width * 0.3), int(height * 0.4) + height + 50), (int(width * 0.3), int(height * 0.7) + height + 50)),  # TO SKIRT AT SIDE
            'E': ((int(width * 0.5), int(height * 0.1) + height + 50), (int(width * 0.5), int(height * 0.9) + height + 50)),  # BACK LENGTH
            'F': ((int(width * 0.2), int(height * 0.2) + height + 50), (int(width * 0.8), int(height * 0.2) + height + 50)),  # SHOULDER TO SHOULDER
            'M': ((int(width * 0.5), int(height * 0.1) + height + 50), (int(width * 0.5), int(height * 0.15) + height + 50)),  # NECKDROP CB
        }
        
        # Calculate and draw front view measurements
        calculated_measurements = {}
        
        for measurement_id, (point1, point2) in front_measurement_points.items():
            # Snap to contour
            idx1 = self.find_closest_point_index(self.contour, point1)
            idx2 = self.find_closest_point_index(self.contour, point2)
            snapped_point1 = tuple(self.contour[idx1])
            snapped_point2 = tuple(self.contour[idx2])
            
            # Calculate measurement
            pixel_distance = self.calculate_measurement(snapped_point1, snapped_point2)
            cm_distance = (pixel_distance / width) * 50
            
            calculated_measurements[measurement_id] = cm_distance
            
            # Draw measurement
            self.draw_technical_measurement_line(annotated, snapped_point1, snapped_point2, 
                                               measurement_id, cm_distance)
        
        # Calculate and draw back view measurements
        for measurement_id, (point1, point2) in back_measurement_points.items():
            # For back view, we need to adjust points to match the mirrored contour
            # This is a simplified approach - in practice you'd need more sophisticated mapping
            original_point1 = (point1[0], point1[1] - height - 50)
            original_point2 = (point2[0], point2[1] - height - 50)
            
            # Snap to original contour
            idx1 = self.find_closest_point_index(self.contour, original_point1)
            idx2 = self.find_closest_point_index(self.contour, original_point2)
            snapped_point1 = tuple(self.contour[idx1])
            snapped_point2 = tuple(self.contour[idx2])
            
            # Calculate measurement
            pixel_distance = self.calculate_measurement(snapped_point1, snapped_point2)
            cm_distance = (pixel_distance / width) * 50
            
            calculated_measurements[measurement_id] = cm_distance
            
            # Draw measurement on back view
            back_point1 = (width - snapped_point1[0], snapped_point1[1] + height + 50)
            back_point2 = (width - snapped_point2[0], snapped_point2[1] + height + 50)
            
            self.draw_technical_measurement_line(annotated, back_point1, back_point2, 
                                               measurement_id, cm_distance)
        
        # Add title and information
        title = "PETRA DRESS - TECHNICAL MEASUREMENTS"
        cv2.putText(annotated, title, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
        
        # Add view labels
        cv2.putText(annotated, "FRONT VIEW", (50, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, "BACK VIEW", (50, height + height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Add measurement legend
        legend_y = canvas_height - 80
        cv2.putText(annotated, "MEASUREMENT LEGEND:", (50, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        legend_items = [
            "A: ½ CHEST", "B: ½ BOTTOM", "C: ARMHOLE DEPTH", "D: TO SKIRT AT SIDE",
            "E: BACK LENGTH", "F: SHOULDER TO SHOULDER", "G: SLANTING SHOULDER", "H: SLEEVE LENGTH",
            "I: BICEPS", "J: BOTTOM SLEEVE", "K: NECKWIDTH", "L: NECKDROP CF",
            "M: NECKDROP CB", "N: NECKLINE EXTENDED"
        ]
        
        for i, item in enumerate(legend_items):
            x = 50 + (i % 4) * 200
            y = legend_y + 25 + (i // 4) * 20
            cv2.putText(annotated, item, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        self.measurements = calculated_measurements
        return annotated, calculated_measurements
    
    def save_final_output(self, annotated_image, measurements):
        """Save final technical drawing"""
        # Save the final technical drawing
        cv2.imwrite("final_annotated_dress.png", annotated_image)
        print("✅ Final technical drawing saved as 'final_annotated_dress.png'")
        
        # Print measurement summary
        print(f"\n=== FINAL DRESS MEASUREMENTS ===")
        for measurement_id, value in measurements.items():
            description = self.measurement_descriptions[measurement_id]
            print(f"{measurement_id}: {description} = {value:.1f}cm")

def main():
    """Main function to create final technical drawing"""
    try:
        # Initialize final annotator
        annotator = FinalDressAnnotator()
        
        # Load and process image
        original_image = annotator.load_and_process_image()
        
        # Create final technical drawing
        annotated_image, measurements = annotator.create_final_technical_drawing()
        
        # Save results
        annotator.save_final_output(annotated_image, measurements)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
