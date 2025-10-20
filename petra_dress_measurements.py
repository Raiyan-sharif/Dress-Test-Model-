import cv2
import numpy as np
import json
from datetime import datetime

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
        """Load image and detect dress contour using simple edge detection (better for line drawings)"""
        print("Loading and processing image...")
        
        # Load image
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get binary image (invert for black lines on white)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            self.contour = max(contours, key=cv2.contourArea)
            self.contour = self.contour.squeeze()
            print(f"Found contour with {len(self.contour)} points")
        else:
            raise ValueError("No contours found in the image.")
        
        return self.image.copy()
    
    def find_closest_point_index(self, contour, point):
        """Find the closest point on contour to given point"""
        point = np.array(point)
        dists = np.sum((contour - point) ** 2, axis=1)
        return np.argmin(dists)
    
    def calculate_measurement(self, point1, point2):
        """Calculate linear measurement between two points"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def determine_dress_size(self, measurements):
        """Determine the closest dress size based on measurements"""
        best_size = None
        min_error = float('inf')
        
        for size, standards in self.size_standards.items():
            total_error = 0
            for measurement_id, measured_value in measurements.items():
                if measurement_id in standards:
                    standard_value = standards[measurement_id]
                    error = abs(measured_value - standard_value)
                    total_error += error
            
            if total_error < min_error:
                min_error = total_error
                best_size = size
        
        return best_size, min_error
    
    def draw_professional_measurement_line(self, img, point1, point2, label, 
                                           color=(0, 0, 0), thickness=2, offset=25):
        """Draw professional measurement line with inward arrows and proper styling"""
        
        # Calculate line properties
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
        
        # Normalize direction vector
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Calculate perpendicular offset for measurement line (flip sign if needed for better placement)
        perp_x = -dy_norm * offset
        perp_y = dx_norm * offset
        
        # Offset points for measurement line
        offset_p1 = (int(point1[0] + perp_x), int(point1[1] + perp_y))
        offset_p2 = (int(point2[0] + perp_x), int(point2[1] + perp_y))
        
        # Draw main measurement line
        cv2.line(img, offset_p1, offset_p2, color, thickness)
        
        # Draw extension lines (from object to measurement line)
        cv2.line(img, point1, offset_p1, color, 1)
        cv2.line(img, point2, offset_p2, color, 1)
        
        # Draw inward arrows
        arrow_length = 12
        arrow_angle = 0.3
        
        # Arrow 1 (inward)
        arrow1_end = (int(offset_p1[0] + dx_norm * arrow_length), 
                      int(offset_p1[1] + dy_norm * arrow_length))
        cv2.arrowedLine(img, offset_p1, arrow1_end, color, thickness, tipLength=arrow_angle)
        
        # Arrow 2 (inward)
        arrow2_end = (int(offset_p2[0] - dx_norm * arrow_length), 
                      int(offset_p2[1] - dy_norm * arrow_length))
        cv2.arrowedLine(img, offset_p2, arrow2_end, color, thickness, tipLength=arrow_angle)
        
        # Draw measurement label (letter only)
        mid_x = (offset_p1[0] + offset_p2[0]) // 2
        mid_y = (offset_p1[1] + offset_p2[1]) // 2
        
        label_text = f"{label}"
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        (label_w, label_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Draw white background for text
        padding = 8
        bg_x1 = mid_x - label_w // 2 - padding
        bg_y1 = mid_y - label_h // 2 - padding
        bg_x2 = mid_x + label_w // 2 + padding
        bg_y2 = mid_y + label_h // 2 + padding
        
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        
        # Draw text
        cv2.putText(img, label_text, (mid_x - label_w // 2, mid_y + label_h // 2), 
                    font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    def calculate_petra_measurements(self):
        """Calculate all PETRA measurements according to specifications"""
        if self.contour is None:
            raise ValueError("No contour available. Run load_and_process_image() first.")
        
        height, width = self.image.shape[:2]
        
        # Create clean white background for professional output
        annotated = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw dress contour in black
        cv2.drawContours(annotated, [self.contour.reshape(-1, 1, 2)], -1, (0, 0, 0), 2)
        
        # Define measurement points based on dress proportions (adjust as needed for accuracy)
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
        
        for measurement_id, (point1, point2) in measurement_points.items():
            # Snap to contour
            idx1 = self.find_closest_point_index(self.contour, point1)
            idx2 = self.find_closest_point_index(self.contour, point2)
            snapped_point1 = tuple(self.contour[idx1])
            snapped_point2 = tuple(self.contour[idx2])
            
            # Calculate measurement (convert pixels to cm - rough approximation assuming ~50cm dress width)
            pixel_distance = self.calculate_measurement(snapped_point1, snapped_point2)
            cm_distance = (pixel_distance / width) * 50
            
            calculated_measurements[measurement_id] = cm_distance
            
            # Draw professional measurement line
            self.draw_professional_measurement_line(annotated, snapped_point1, snapped_point2, measurement_id)
        
        # Add title
        title = "PETRA DRESS - TECHNICAL MEASUREMENTS"
        cv2.putText(annotated, title, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        
        self.measurements = calculated_measurements
        return annotated, calculated_measurements
    
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
        cv2.imwrite("annotated_dress.png", annotated_image)
        print("✅ Professional annotated image saved as 'annotated_dress.png'")
        
        # Save measurement report
        with open("measurement_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("✅ Measurement report saved as 'measurement_report.json'")
        
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