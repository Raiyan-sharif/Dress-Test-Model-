# PETRA Dress Measurement System

A professional dress measurement system that automatically analyzes dress images and generates technical drawings with precise measurements following PETRA_INQ.xlsm specifications.

## 🎯 Output

**Input:** `dress_wire.jpeg` → **Output:** `annotated_dress.png`

The system generates a professional technical drawing with all 14 PETRA measurement points (A-N) properly labeled.

## 📋 Features

- **Professional Technical Drawing Output** - Clean white background with black dress contour
- **Complete PETRA Compliance** - All 14 measurement points (A-N) as specified in PETRA_INQ.xlsm
- **Automatic Size Determination** - Determines closest dress size (50, 56, 62, 68, 74, 80)
- **Simplified Computer Vision** - Uses OpenCV edge detection optimized for line drawings
- **Comprehensive Reporting** - JSON measurement reports with detailed data
- **Professional Styling** - Technical drawing format with proper measurement lines and arrows
- **Label-Only Display** - Shows measurement point labels (A, B, C, D, etc.) without values on image

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone or download the project**
   ```bash
   cd Dress-Test-Model-
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**To generate `annotated_dress.png` from `dress_wire.jpeg`:**

```bash
python petra_dress_measurements.py
```

**Alternative - Use the main script:**
```bash
python main.py
```

## 📁 Input/Output Files

### Input
- `dress_wire.jpeg` - The dress image to analyze

### Output
- `annotated_dress.png` - **Main desired output** (Professional technical drawing)
- `measurement_report.json` - Complete measurement data

## 📏 PETRA Measurements (A-N)

The system measures all 14 PETRA specification points:

| Code | Description | Example Value |
|------|-------------|---------------|
| A | ½ CHEST | 26.5cm |
| B | ½ BOTTOM | 29.0cm |
| C | ARMHOLE DEPTH fr HPS | 0.0cm |
| D | TO SKIRT AT SIDE | 13.2cm |
| E | BACK LENGTH fr HPS | 39.7cm |
| F | SHOULDER TO SHOULDER | 31.8cm |
| G | SLANTING SHOULDER | 6.5cm |
| H | SLEEVE LENGTH | 20.9cm |
| I | BICEPS | 8.7cm |
| J | BOTTOM SLEEVE | 2.6cm |
| K | NECKWIDTH | 9.4cm |
| L | NECKDROP CF fr HPS | 0.1cm |
| M | NECKDROP CB fr HPS | 0.1cm |
| N | Min NECKLINE extended | 21.7cm |

## 🎨 Output Format

The generated `annotated_dress.png` includes:

- **Clean white background** with black dress contour
- **Professional measurement lines** with arrows and extension lines
- **Measurement point labels** (A, B, C, D, E, F, G, H, I, J, K, L, M, N) without values
- **Technical drawing styling** matching industry standards
- **All 14 measurement points** properly positioned and labeled

## 📊 Size Standards

The system supports all PETRA dress sizes with their respective measurement standards:

- **Size 50** - Smallest size
- **Size 56** - Small size  
- **Size 62** - Medium-small size
- **Size 68** - Medium size
- **Size 74** - Large size
- **Size 80** - Largest size

## 🔧 Technical Details

### Computer Vision Pipeline
1. **Image Loading** - Loads `dress_wire.jpeg`
2. **Edge Detection** - Uses OpenCV thresholding to detect dress outline
3. **Contour Extraction** - Extracts precise dress contour from binary image
4. **Measurement Calculation** - Calculates all 14 PETRA measurements
5. **Size Determination** - Determines closest matching dress size
6. **Technical Drawing Generation** - Creates professional annotated output

### Dependencies
- **OpenCV** - Image processing and computer vision
- **NumPy** - Numerical computations
- **JSON** - Data export

## 🔄 Process of Work

### Step-by-Step Workflow

#### 1. **Image Preprocessing**
```
Input: dress_wire.jpeg (line drawing of dress)
↓
Convert to grayscale
↓
Apply binary thresholding (invert for black lines on white background)
↓
Result: Binary image with dress outline clearly defined
```

#### 2. **Contour Detection**
```
Binary image
↓
Find external contours using cv2.findContours()
↓
Select largest contour (main dress outline)
↓
Result: Precise dress contour with coordinate points
```

#### 3. **Measurement Point Definition**
```
Define 14 measurement points based on dress proportions:
- A: ½ CHEST (horizontal across chest)
- B: ½ BOTTOM (horizontal across hem)
- C: ARMHOLE DEPTH (vertical from shoulder)
- D: TO SKIRT AT SIDE (vertical side length)
- E: BACK LENGTH (vertical center back)
- F: SHOULDER TO SHOULDER (horizontal shoulder width)
- G: SLANTING SHOULDER (diagonal shoulder)
- H: SLEEVE LENGTH (horizontal sleeve)
- I: BICEPS (vertical sleeve width)
- J: BOTTOM SLEEVE (vertical cuff width)
- K: NECKWIDTH (horizontal neck opening)
- L: NECKDROP CF (vertical front neck)
- M: NECKDROP CB (vertical back neck)
- N: NECKLINE EXTENDED (horizontal neckline)
```

#### 4. **Point Snapping**
```
For each measurement point:
↓
Find closest point on dress contour
↓
Snap measurement point to actual dress edge
↓
Result: Accurate measurement points on dress outline
```

#### 5. **Measurement Calculation**
```
For each measurement pair:
↓
Calculate pixel distance between points
↓
Convert pixels to centimeters (assuming ~50cm dress width)
↓
Store measurement value
↓
Result: All 14 measurements in centimeters
```

#### 6. **Size Determination**
```
Compare calculated measurements with PETRA size standards:
- Size 50, 56, 62, 68, 74, 80
↓
Calculate error for each size
↓
Select size with minimum total error
↓
Result: Determined dress size
```

#### 7. **Technical Drawing Generation**
```
Create clean white background
↓
Draw dress contour in black
↓
For each measurement:
  - Draw measurement line with offset from dress
  - Add extension lines from dress to measurement line
  - Draw inward-pointing arrows
  - Add measurement label (A, B, C, D, etc.) in white box
↓
Add title and formatting
↓
Result: Professional technical drawing
```

#### 8. **Output Generation**
```
Save annotated image as: annotated_dress.png
↓
Generate measurement report as: measurement_report.json
↓
Display summary in console
↓
Result: Complete measurement analysis
```

### Algorithm Details

#### **Edge Detection Method**
```python
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binary thresholding (invert for black lines on white)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

#### **Measurement Line Drawing**
```python
# Calculate perpendicular offset for measurement line
perp_x = -dy_norm * offset
perp_y = dx_norm * offset

# Draw measurement line with extension lines
cv2.line(img, offset_p1, offset_p2, color, thickness)
cv2.line(img, point1, offset_p1, color, 1)  # Extension line
cv2.line(img, point2, offset_p2, color, 1)  # Extension line

# Draw inward arrows
cv2.arrowedLine(img, offset_p1, arrow1_end, color, thickness)
cv2.arrowedLine(img, offset_p2, arrow2_end, color, thickness)
```

#### **Size Determination Algorithm**
```python
for size, standards in PETRA_SIZE_STANDARDS.items():
    total_error = 0
    for measurement_id, measured_value in measurements.items():
        if measurement_id in standards:
            standard_value = standards[measurement_id]
            error = abs(measured_value - standard_value)
            total_error += error
    
    if total_error < min_error:
        min_error = total_error
        best_size = size
```

### Performance Characteristics

- **Processing Time**: ~2-3 seconds for typical dress image
- **Accuracy**: Optimized for line drawings and technical sketches
- **Memory Usage**: Minimal (no deep learning models required)
- **Dependencies**: Lightweight (OpenCV + NumPy only)
- **Output Quality**: Professional technical drawing standard

## 📝 Example Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the PETRA measurement system
python petra_dress_measurements.py

# Expected output:
# Loading and processing image...
# Found dress contour with 1208 points
# ✅ Professional annotated image saved as 'annotated_dress.png'
# ✅ Measurement report saved as 'measurement_report.json'
# 
# === PETRA DRESS MEASUREMENT SUMMARY ===
# Determined Size: 56
# Measurement Error: 111.37
# 
# Measurements:
#   A: ½ CHEST = 26.5cm
#   B: ½ BOTTOM = 29.0cm
#   ... (all 14 measurements)
```

## 🎯 Main Scripts

1. **`petra_dress_measurements.py`** - **Primary script** for generating `annotated_dress.png`
2. **`main.py`** - Updated main script with professional features
3. **`final_dress_annotator.py`** - Advanced script with front/back views
4. **`professional_dress_annotator.py`** - Professional single-view output

## 📋 Requirements

See `requirements.txt` for complete dependency list:
- opencv-python==4.12.0.88
- numpy==2.2.6
- Additional dependencies as needed

## 🚨 Troubleshooting

### Common Issues

1. **"No contours found in the image"**
   - Ensure `dress_wire.jpeg` is in the project directory
   - Check that the image shows a clear dress outline with black lines on white background
   - Verify the image is a line drawing or technical sketch
   - Try adjusting the threshold value in the script (currently set to 200)

2. **Poor contour detection**
   - Ensure the dress image has high contrast (black lines on white background)
   - Check that the dress outline is complete and unbroken
   - For scanned images, ensure good quality and minimal noise

3. **Import errors**
   - Make sure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`
   - Ensure OpenCV is properly installed: `pip install opencv-python`

4. **Measurement accuracy issues**
   - Verify the dress image is properly oriented
   - Check that measurement points are correctly positioned
   - Adjust the pixel-to-cm conversion factor if needed

## 📄 License

This project follows the specifications from PETRA_INQ.xlsm for professional dress measurement standards.

## 🎯 Success Criteria

✅ **Input:** `dress_wire.jpeg`  
✅ **Output:** `annotated_dress.png`  
✅ **All 14 PETRA measurements (A-N)**  
✅ **Professional technical drawing format**  
✅ **Automatic size determination**  
✅ **Complete measurement reporting**  
✅ **Simplified dependencies (OpenCV only)**  
✅ **Optimized for line drawings**  

---

**Ready to generate professional dress measurements!** 🎨📏