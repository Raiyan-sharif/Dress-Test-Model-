# PETRA Dress Measurement System

A professional dress measurement system that automatically analyzes dress images and generates technical drawings with precise measurements following PETRA_INQ.xlsm specifications.

## 🎯 Output

**Input:** `dress_wire.jpeg` → **Output:** `petra_annotated_dress.png`

The system generates a professional technical drawing with all 14 PETRA measurements (A-N) properly labeled and measured.

## 📋 Features

- **Professional Technical Drawing Output** - Clean white background with black dress contour
- **Complete PETRA Compliance** - All 14 measurement points (A-N) as specified in PETRA_INQ.xlsm
- **Automatic Size Determination** - Determines closest dress size (50, 56, 62, 68, 74, 80)
- **Advanced Computer Vision** - Uses detectron2 Mask R-CNN for precise dress detection
- **Comprehensive Reporting** - JSON measurement reports with detailed data
- **Professional Styling** - Technical drawing format with proper measurement lines and arrows

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

4. **Install detectron2 and dependencies**
   ```bash
   pip install fvcore black cloudpickle hydra-core omegaconf pycocotools tensorboard
   pip install "iopath>=0.1.7,<0.1.10"
   ```

### Usage

**To generate `petra_annotated_dress.png` from `dress_wire.jpeg`:**

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
- `petra_annotated_dress.png` - **Main desired output** (Professional technical drawing)
- `petra_measurement_report.json` - Complete measurement data
- `annotated_dress.png` - Alternative output from main.py
- `measurement_report.json` - Alternative measurement report

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

The generated `petra_annotated_dress.png` includes:

- **Clean white background** with black dress contour
- **Professional measurement lines** with arrows and extension lines
- **Measurement labels** with values in centimeters
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
2. **Object Detection** - Uses detectron2 Mask R-CNN to detect dress
3. **Contour Extraction** - Extracts precise dress outline
4. **Measurement Calculation** - Calculates all 14 PETRA measurements
5. **Size Determination** - Determines closest matching dress size
6. **Technical Drawing Generation** - Creates professional annotated output

### Dependencies
- **detectron2** - Advanced computer vision framework
- **OpenCV** - Image processing
- **PyTorch** - Deep learning backend
- **NumPy** - Numerical computations
- **JSON** - Data export

## 📝 Example Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the PETRA measurement system
python petra_dress_measurements.py

# Expected output:
# Loading and processing image...
# Loading Mask R-CNN model...
# Number of instances detected: 3
# Found dress contour with 1208 points
# ✅ Annotated image saved as 'petra_annotated_dress.png'
# ✅ Measurement report saved as 'petra_measurement_report.json'
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

1. **`petra_dress_measurements.py`** - **Primary script** for generating `petra_annotated_dress.png`
2. **`main.py`** - Updated main script with professional features
3. **`final_dress_annotator.py`** - Advanced script with front/back views
4. **`professional_dress_annotator.py`** - Professional single-view output

## 📋 Requirements

See `requirements.txt` for complete dependency list:
- torch==2.9.0
- torchvision==0.24.0
- opencv-python==4.12.0.88
- numpy==2.2.6
- detectron2 (installed separately)
- Additional dependencies for detectron2

## 🚨 Troubleshooting

### Common Issues

1. **"detectron2 is not installed"**
   ```bash
   pip install fvcore black cloudpickle hydra-core omegaconf pycocotools tensorboard
   pip install "iopath>=0.1.7,<0.1.10"
   ```

2. **"No instances detected"**
   - Ensure `dress_wire.jpeg` is in the project directory
   - Check that the image shows a clear dress outline
   - Try adjusting the detection threshold in the script

3. **Import errors**
   - Make sure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

## 📄 License

This project follows the specifications from PETRA_INQ.xlsm for professional dress measurement standards.

## 🎯 Success Criteria

✅ **Input:** `dress_wire.jpeg`  
✅ **Output:** `petra_annotated_dress.png`  
✅ **All 14 PETRA measurements (A-N)**  
✅ **Professional technical drawing format**  
✅ **Automatic size determination**  
✅ **Complete measurement reporting**

---

**Ready to generate professional dress measurements!** 🎨📏