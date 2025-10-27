# Enhanced CubiCasa5K Dataset Processing

This directory contains enhanced processing tools for the CubiCasa5K dataset with proper train/test splitting and comprehensive image processing.

## 🚀 Quick Start

### Process Full Dataset (Recommended)
```bash
# Process the complete dataset with 60/40 train/test split
python process_full_dataset.py
```

### Process Sample Dataset (For Testing)
```bash
# Process only 50 samples for quick testing
python data/process_cubicasa5k_improved.py --max-samples 50 --train-ratio 0.6
```

### Interactive Processing (Jupyter Notebook)
```bash
# Launch Jupyter and open the processing notebook
jupyter notebook notebooks/CubiCasa5K_Processing_Enhanced.ipynb
```

## 📁 Dataset Structure

The CubiCasa5K dataset should be organized as follows:
```
data/
├── cubicasa5k/
│   ├── high_quality/
│   │   ├── 10004/
│   │   │   ├── F1_original.png
│   │   │   ├── F1_scaled.png
│   │   │   └── model.svg
│   │   └── ...
│   ├── high_quality_architectural/
│   │   └── ...
│   └── colorful/
│       └── ...
└── processed/
    ├── images/           # Processed 512x512 images
    ├── cubicasa5k_full.csv
    ├── cubicasa5k_train.csv
    └── cubicasa5k_test.csv
```

## 🔧 Processing Features

### Enhanced Image Processing
- **Smart Resizing**: Maintains aspect ratio with white padding
- **Quality Enhancement**: Applies contrast and sharpness improvements
- **Standardized Size**: All images resized to 512×512 pixels
- **Format Consistency**: All images saved as PNG with RGB mode

### Proper Train/Test Splitting
- **60/40 Split**: 60% training, 40% testing (configurable)
- **Stratified Splitting**: Attempts to balance room count distribution
- **Reproducible**: Uses fixed random seed for consistent splits
- **No Data Leakage**: Ensures proper separation between train and test

### Comprehensive Metadata Extraction
- **Room Analysis**: Counts rooms and identifies types
- **Architectural Features**: Detects balconies, garages, etc.
- **Style Classification**: Infers architectural styles
- **Area Estimation**: Estimates floor plan area
- **Quality Metrics**: Tracks image dimensions and file sizes

### Rich Descriptions
- **Natural Language**: Generates human-readable descriptions
- **Feature-Rich**: Includes room counts, styles, and special features
- **Category Context**: Adds dataset category information
- **Structured Format**: Consistent description patterns

## 📊 Generated Outputs

### CSV Files
- `cubicasa5k_full.csv` - Complete processed dataset
- `cubicasa5k_train.csv` - Training set (60%)
- `cubicasa5k_test.csv` - Test set (40%)
- `metadata.csv` - Training pipeline compatible format

### Images
- `processed/images/` - All processed 512×512 images
- Naming: `{category}_{index:05d}_{original_id}.png`

### Statistics and Analysis
- `cubicasa5k_enhanced_statistics.json` - Detailed statistics
- `cubicasa5k_enhanced_analysis.png` - Comprehensive visualization
- Split files: `train.txt`, `test.txt` for compatibility

## 🎯 Key Improvements Over Original

### 1. Proper Dataset Splitting
- **Before**: Random 84%/8%/8% split with potential data leakage
- **After**: Stratified 60%/40% split with proper separation

### 2. Enhanced Image Processing
- **Before**: Basic resizing without aspect ratio preservation
- **After**: Smart resizing with padding and quality enhancement

### 3. Rich Metadata
- **Before**: Basic room counting
- **After**: Comprehensive feature extraction and natural descriptions

### 4. Better Organization
- **Before**: Mixed processing approaches
- **After**: Unified pipeline with clear structure

### 5. Comprehensive Analysis
- **Before**: Basic statistics
- **After**: Multi-dimensional analysis with visualizations

## 🔍 Dataset Statistics (Example)

```
📊 Dataset Overview
==================
Total samples: 4,997
Train samples: 2,998 (60.0%)
Test samples: 1,999 (40.0%)
Categories: 4 (high_quality, high_quality_architectural, colorful, root)
Average room count: 3.2 ± 1.8
Average area: 520 ± 180 sq ft
Architectural styles: modern, contemporary, traditional
Special features: 1,499 balconies (30%), 1,999 garages (40%)
```

## 🛠️ Configuration Options

### Command Line Arguments
```bash
python data/process_cubicasa5k_improved.py \
    --data-dir /path/to/data \
    --train-ratio 0.6 \
    --max-samples 1000 \
    --target-size 512 512
```

### Parameters
- `--data-dir`: Data directory path (default: current directory)
- `--train-ratio`: Training set ratio (default: 0.6 for 60%)
- `--max-samples`: Maximum samples to process (default: None for all)
- `--target-size`: Target image size as width height (default: 512 512)

## 📈 Quality Metrics

### Image Quality
- **Resolution**: Standardized 512×512 pixels
- **Aspect Ratio**: Preserved with intelligent padding
- **Enhancement**: Improved contrast and sharpness
- **Format**: Consistent PNG/RGB format

### Dataset Quality
- **Balance**: Stratified splitting maintains distribution
- **Coverage**: All categories and styles represented
- **Completeness**: Comprehensive metadata for all samples
- **Consistency**: Uniform processing across all images

## 🚀 Integration with Training Pipeline

The processed dataset is fully compatible with the FloorMind training pipeline:

```python
# Load processed dataset
import pandas as pd
df = pd.read_csv('data/processed/cubicasa5k_train.csv')

# Access processed images
from PIL import Image
image = Image.open(df.iloc[0]['image_path'])

# Use rich descriptions
description = df.iloc[0]['description']
# "Modern 3-room apartment featuring living room, bedroom, kitchen with balcony"
```

## 🔧 Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   FileNotFoundError: CubiCasa5K dataset not found
   ```
   - Ensure the dataset is extracted to `data/cubicasa5k/`
   - Check directory structure matches expected format

2. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   - Use `--max-samples` to process smaller batches
   - Ensure sufficient disk space for processed images

3. **Stratification Failed**
   ```
   ⚠️ Stratification failed, using random split
   ```
   - Normal for small datasets or uneven distributions
   - Random split is used as fallback

### Performance Tips

- **Parallel Processing**: The script processes images sequentially for stability
- **Disk Space**: Ensure ~2GB free space for full dataset processing
- **Memory**: Recommended 8GB+ RAM for full dataset processing

## 📝 Next Steps

After processing, you can:

1. **Train Models**: Use the processed dataset with FloorMind training pipeline
2. **Analyze Results**: Review the generated statistics and visualizations
3. **Customize Processing**: Modify the script for specific requirements
4. **Extend Features**: Add new metadata extraction or processing steps

## 🤝 Contributing

To improve the processing pipeline:

1. **Add New Features**: Extend metadata extraction functions
2. **Improve Quality**: Enhance image processing algorithms
3. **Add Formats**: Support additional image formats or datasets
4. **Optimize Performance**: Implement parallel processing or caching

## 📚 References

- [CubiCasa5K Dataset](https://github.com/CubiCasa/CubiCasa5k)
- [FloorMind Training Pipeline](../training/)
- [Image Processing Best Practices](../docs/image_processing.md)