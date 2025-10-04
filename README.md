# Leaf Detection

Leaf Detection is a computer vision project aimed at recognizing and analyzing leaves from digital images. This repository provides an automated pipeline for detecting, segmenting, and extracting features from images containing leaves. The solution leverages classical image processing techniques and machine learning models, making it suitable for plant species identification, disease detection, and agricultural research.

---

## Features

- Automated leaf detection and segmentation from images.
- Extraction of morphological and color features.
- Preprocessing pipeline for noise reduction and background removal.
- Machine learning model integration for classification tasks.
- Visualization tools for input images, segmentation masks, and classification output.
- Modular code structure for easy extension and customization.

---

## Requirements

To successfully run the Leaf Detection project, ensure your environment meets the following requirements:

- Python 3.6 or higher
- OpenCV (cv2)
- NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook (for interactive exploration)
- Additional dependencies as listed in the `requirements.txt` (if present)

---

## Installation

Follow these steps to set up the Leaf Detection project:

1. **Clone the repository**
   ```bash
   git clone https://github.com/suyashambule/Leaf-detection-.git
   cd Leaf-detection-
   ```

2. **Set up a Python virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   If a `requirements.txt` is not provided, install dependencies manually:
   ```bash
   pip install opencv-python numpy scikit-learn matplotlib
   ```

---

## Configuration

Before running the system, configure the following settings as needed:

- **Input Directory**: Place your leaf images in the `input` or designated images folder.
- **Output Directory**: Set the path for saving processed images and results, if configurable.
- **Model Parameters**: Adjust any machine learning parameters or thresholds in the configuration files or within the scripts.
- **Preprocessing**: Tweak preprocessing parameters such as filter kernel size or segmentation thresholds for optimal results on your dataset.

Configuration can usually be modified in a configuration file (e.g., `config.py`) or directly in the main script or notebook.

---

## Usage

### Basic Workflow

1. **Prepare your dataset**
   - Gather leaf images and save them in the designated input folder.

2. **Run the detection pipeline**
   - Execute the main script or Jupyter notebook provided in the repository.
   - Example command:
     ```bash
     python main.py --input_dir ./input --output_dir ./output
     ```
   - Or open the notebook:
     ```bash
     jupyter notebook LeafDetection.ipynb
     ```

3. **View results**
   - Segmented images, extracted features, and classification results will be saved in the output directory.
   - Visualizations are available within the notebook or as image files.

### Example: Leaf Segmentation and Feature Extraction

```python
import cv2
import numpy as np
from leaf_detection import segment_leaf, extract_features

img = cv2.imread('input/leaf_sample.jpg')
mask = segment_leaf(img)
features = extract_features(mask, img)
print(features)
```

### API and Functionality Overview

The following Mermaid diagram illustrates the high-level workflow for leaf detection and analysis:

```mermaid
flowchart TD
    A[Input Leaf Image] --> B[Preprocessing]
    B --> C[Segmentation]
    C --> D[Feature Extraction]
    D --> E[Classification (Optional)]
    E --> F[Visualization & Output]
```

- **Preprocessing**: Resizing, denoising, and color normalization.
- **Segmentation**: Isolating leaf from background using thresholding or edge detection.
- **Feature Extraction**: Computing shape, color, and texture features.
- **Classification**: (Optional) Predicting plant species or leaf condition.
- **Visualization**: Displaying original and processed results.

---

## Contributing

Contributions are welcome! To contribute:

- Fork the repository.
- Create a new branch for your feature or bugfix.
- Commit your changes with descriptive messages.
- Open a pull request describing your modifications.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or support, please open an issue in the repository or contact the maintainer via their GitHub profile.

---

Enjoy automated leaf detection and classification with this project! Contributions and feedback are highly appreciated.